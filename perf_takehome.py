"""
# Anthropic's Original Performance Engineering Take-home (Release version)

Copyright Anthropic PBC 2026. Permission is granted to modify and use, but not
to publish or redistribute your solutions so it's hard to find spoilers.

# Task

- Optimize the kernel (in KernelBuilder.build_kernel) as much as possible in the
  available time, as measured by test_kernel_cycles on a frozen separate copy
  of the simulator.

Validate your results using `python tests/submission_tests.py` without modifying
anything in the tests/ folder.

We recommend you look through problem.py next.
"""

from collections import defaultdict
import random
import unittest

from problem import (
    Engine,
    DebugInfo,
    SLOT_LIMITS,
    VLEN,
    N_CORES,
    SCRATCH_SIZE,
    Machine,
    Tree,
    Input,
    HASH_STAGES,
    reference_kernel,
    build_mem_image,
    reference_kernel2,
)

class KernelBuilder:
    def __init__(self):
        self.instrs = []
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.const_map = {}
        self.vconst_map = {}
        self.enable_debug = True

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def _slot_reads_writes(self, engine, slot):
        reads = set()
        writes = set()
        barrier = False

        def add_vec(base):
            return set(range(base, base + VLEN))

        if engine == "debug":
            return reads, writes, True

        if engine == "alu":
            _, dest, a1, a2 = slot
            reads.update([a1, a2])
            writes.add(dest)
        elif engine == "valu":
            op = slot[0]
            if op == "vbroadcast":
                _, dest, src = slot
                reads.add(src)
                writes.update(add_vec(dest))
            elif op == "multiply_add":
                _, dest, a, b, c = slot
                reads.update(add_vec(a))
                reads.update(add_vec(b))
                reads.update(add_vec(c))
                writes.update(add_vec(dest))
            else:
                _, dest, a1, a2 = slot
                reads.update(add_vec(a1))
                reads.update(add_vec(a2))
                writes.update(add_vec(dest))
        elif engine == "load":
            match slot:
                case ("load", dest, addr):
                    reads.add(addr)
                    writes.add(dest)
                case ("load_offset", dest, addr, offset):
                    reads.add(addr + offset)
                    writes.add(dest + offset)
                case ("vload", dest, addr):
                    reads.add(addr)
                    writes.update(add_vec(dest))
                case ("const", dest, _val):
                    writes.add(dest)
        elif engine == "store":
            match slot:
                case ("store", addr, src):
                    reads.update([addr, src])
                case ("vstore", addr, src):
                    reads.add(addr)
                    reads.update(add_vec(src))
        elif engine == "flow":
            op = slot[0]
            if op in ("halt", "pause", "jump", "jump_indirect", "cond_jump", "cond_jump_rel"):
                barrier = True
            match slot:
                case ("select", dest, cond, a, b):
                    reads.update([cond, a, b])
                    writes.add(dest)
                case ("add_imm", dest, a, _imm):
                    reads.add(a)
                    writes.add(dest)
                case ("vselect", dest, cond, a, b):
                    reads.update(add_vec(cond))
                    reads.update(add_vec(a))
                    reads.update(add_vec(b))
                    writes.update(add_vec(dest))
                case ("trace_write", val):
                    reads.add(val)
                case ("coreid", dest):
                    writes.add(dest)

        return reads, writes, barrier


    def build(self, slots, vliw=False):
        # Improved slot packing with look-ahead
        instrs = []
        used = [False] * len(slots)
        n = len(slots)
        
        bundle = {}
        bundle_input = set()
        bundle_output = set()
        
        def flush():
            nonlocal bundle, bundle_input, bundle_output
            if bundle:
                instrs.append(bundle)
            bundle = {}
            bundle_input = set()
            bundle_output = set()
        
        def can_add_to_bundle(engine, slot_input, slot_output):
            """Check if slot can be added to current bundle without conflicts"""
            # Check engine slot limit
            if len(bundle.get(engine, [])) >= SLOT_LIMITS[engine]:
                return False
            # Check if slot reads from bundle outputs (RAW dependency - must flush)
            if slot_input & bundle_output:
                return False
            # Check if slot writes to bundle outputs (WAW dependency - must flush)
            if slot_output & bundle_output:
                return False
            # WAR dependencies within a bundle are OK in VLIW since reads happen
            # before writes in the same cycle
            return True

        i = 0
        while i < n:
            if used[i]:
                i += 1
                continue
                
            engine, slot = slots[i]
            input, output, barrier = self._slot_reads_writes(engine, slot)
            
            if barrier:
                flush()
                instrs.append({engine: [slot]})
                used[i] = True
                i += 1
                continue
            
            # Try to add current slot to bundle
            if can_add_to_bundle(engine, input, output):
                bundle.setdefault(engine, []).append(slot)
                bundle_input.update(input)
                bundle_output.update(output)
                used[i] = True
                i += 1
                # Conservative look-ahead: only add slots that don't conflict
                # Track what skipped slots READ and WRITE to avoid reordering issues
                skipped_outputs = set()  # Track what skipped slots WRITE
                for look_ahead in range(i, min(n, i + 100)):
                    if used[look_ahead]:
                        continue
                    next_engine, next_slot = slots[look_ahead]
                    next_input, next_output, next_barrier = self._slot_reads_writes(next_engine, next_slot)
                    if next_barrier:
                        break
                    # Can't add if slot reads from skipped slot's output (RAW - would execute too early)
                    if next_input & skipped_outputs:
                        skipped_outputs.update(next_output)
                        continue
                    # Can't add if slot writes to skipped slot's output (WAW - would overwrite wrong value)
                    if next_output & skipped_outputs:
                        skipped_outputs.update(next_output)
                        continue
                    if can_add_to_bundle(next_engine, next_input, next_output):
                        bundle.setdefault(next_engine, []).append(next_slot)
                        bundle_input.update(next_input)
                        bundle_output.update(next_output)
                        used[look_ahead] = True
                    else:
                        skipped_outputs.update(next_output)
            else:
                # Can't add, flush and add to new bundle
                flush()
                bundle.setdefault(engine, []).append(slot)
                bundle_input.update(input)
                bundle_output.update(output)
                used[i] = True
                i += 1
        
        flush()
        return instrs

    def add(self, engine, slot):
        self.instrs.append({engine: [slot]})

    def alloc_scratch(self, name=None, length=1):
        addr = self.scratch_ptr
        if name is not None:
            self.scratch[name] = addr
            self.scratch_debug[addr] = (name, length)
        self.scratch_ptr += length
        assert self.scratch_ptr <= SCRATCH_SIZE, "Out of scratch space"
        return addr

    def scratch_const(self, val, name=None, init_slots=None):
        if val not in self.const_map:
            addr = self.alloc_scratch(name)
            if init_slots is not None:
                init_slots.append(("load", ("const", addr, val)))
            else:
                self.add("load", ("const", addr, val))
            self.const_map[val] = addr
        return self.const_map[val]
    
    def scratch_vconst(self, val, name=None, init_slots=None):
        if val not in self.vconst_map:
            addr = self.alloc_scratch(name, length=VLEN)
            scalar_addr = self.scratch_const(val, init_slots=init_slots)
            if init_slots is not None:
                init_slots.append(("valu", ("vbroadcast", addr, scalar_addr)))
            else:
                self.add("valu", ("vbroadcast", addr, scalar_addr))
            self.vconst_map[val] = addr
        return self.vconst_map[val]

    def build_hash(self, val_hash_addr, tmp1, tmp2, round, i):
        slots = []

        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            slots.append(("alu", (op1, tmp1, val_hash_addr, self.scratch_const(val1))))
            slots.append(("alu", (op3, tmp2, val_hash_addr, self.scratch_const(val3))))
            slots.append(("alu", (op2, val_hash_addr, tmp1, tmp2)))
            if self.enable_debug:
                slots.append(("debug", ("compare", val_hash_addr, (round, i, "hash_stage", hi))))

        return slots

    def build_vhash(self, val_hash_addr, tmp1, tmp2, round, i):
        slots = []

        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            if op1 == "+" and op2 == "+" and op3 == "<<":
                # Use multiply_add
                factor = 1 + (1 << val3)
                slots.append(("valu", ("multiply_add", val_hash_addr, val_hash_addr, self.scratch_vconst(factor), self.scratch_vconst(val1))))
            else:
                slots.append(("valu", (op1, tmp1, val_hash_addr, self.scratch_vconst(val1))))
                slots.append(("valu", (op3, tmp2, val_hash_addr, self.scratch_vconst(val3))))
                slots.append(("valu", (op2, val_hash_addr, tmp1, tmp2)))
            if self.enable_debug:
                slots.append(("debug", ("vcompare", val_hash_addr, [(round, i+idx, "hash_stage", hi) for idx in range(VLEN)])))
        
        return slots

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """
        Optimized implementation with multi-block interleaving for maximum parallelism.
        Process 2 blocks simultaneously to overlap scatter loads with valu computation.
        """
        num_blocks = batch_size // VLEN
        
        # Scratch space addresses
        init_vars = [
            "rounds",
            "n_nodes",
            "batch_size",
            "forest_height",
            "forest_values_p",
            "inp_indices_p",
            "inp_values_p",
        ]
        for v in init_vars:
            self.alloc_scratch(v, 1)
        # Collect initialization slots for bundling
        init_slots = []
        
        # Use vload to load all header values at once (7 values, VLEN=8)
        header_base = self.alloc_scratch("header", VLEN)
        zero_addr = self.scratch_const(0, init_slots=init_slots)
        init_slots.append(("load", ("vload", header_base, zero_addr)))
        # Map init_vars to their positions in the header block
        for i, v in enumerate(init_vars):
            self.scratch[v] = header_base + i
            self.scratch_debug[header_base + i] = (v, 1)

        # Precompute pointer addresses for each block
        idx_ptrs = self.alloc_scratch("idx_ptrs", batch_size // VLEN)
        val_ptrs = self.alloc_scratch("val_ptrs", batch_size // VLEN)
        
        # Use alu to compute: idx_ptrs[b] = inp_indices_p + b*VLEN
        for b in range(batch_size // VLEN):
            offset = self.scratch_const(b * VLEN)
            init_slots.append(("alu", ("+", idx_ptrs + b, self.scratch["inp_indices_p"], offset)))
            init_slots.append(("alu", ("+", val_ptrs + b, self.scratch["inp_values_p"], offset)))

        # Pre-allocate ALL constants that will be needed during hash computation
        # This allows them to be bundled together during init
        vone_const = self.scratch_vconst(1, "vone", init_slots=init_slots)
        vtwo_const = self.scratch_vconst(2, "vtwo", init_slots=init_slots)
        
        # Pre-allocate hash stage constants
        for stage_idx in range(len(HASH_STAGES)):
            op1, val1, op2, op3, val3 = HASH_STAGES[stage_idx]
            if op1 == "+" and op2 == "+" and op3 == "<<":
                factor = 1 + (1 << val3)
                self.scratch_vconst(factor, init_slots=init_slots)
                self.scratch_vconst(val1, init_slots=init_slots)
            else:
                self.scratch_vconst(val1, init_slots=init_slots)
                self.scratch_vconst(val3, init_slots=init_slots)

        # Allocate PERSISTENT scratch for ALL blocks' idx and val
        # This allows us to keep values across rounds without reloading
        persistent_scratch = []  # [block_idx] -> {idx, val}
        for b in range(num_blocks):
            persistent_scratch.append({
                'idx': self.alloc_scratch(f"block_{b}_idx", VLEN),
                'val': self.alloc_scratch(f"block_{b}_val", VLEN),
            })
        
        # Allocate temporary scratch for processing groups (reused across groups)
        NUM_PARALLEL_BLOCKS = 4
        NUM_GROUPS = 2
        group_temps = []  # [group][local_idx] -> {node_val, addr, tmp1, tmp2, tmp3}
        for g in range(NUM_GROUPS):
            temps = []
            for b in range(NUM_PARALLEL_BLOCKS):
                temps.append({
                    'node_val': self.alloc_scratch(f"g{g}_node_val_{b}", VLEN),
                    'addr': self.alloc_scratch(f"g{g}_addr_{b}", VLEN),
                    'tmp1': self.alloc_scratch(f"g{g}_tmp1_{b}", VLEN),
                    'tmp2': self.alloc_scratch(f"g{g}_tmp2_{b}", VLEN),
                    'tmp3': self.alloc_scratch(f"g{g}_tmp3_{b}", VLEN),
                })
            group_temps.append(temps)
        
        v_n_nodes = self.alloc_scratch("v_n_nodes", VLEN)
        init_slots.append(("valu", ("vbroadcast", v_n_nodes, self.scratch["n_nodes"])))
        v_forest_p = self.alloc_scratch("v_forest_p", VLEN)
        init_slots.append(("valu", ("vbroadcast", v_forest_p, self.scratch["forest_values_p"])))
        
        # Bundle initialization slots
        init_instrs = self.build(init_slots)
        self.instrs.extend(init_instrs)

        self.add("flow", ("pause",))
        if self.enable_debug:
            self.add("debug", ("comment", "Starting loop"))

        body = []
        
        num_groups = (num_blocks + NUM_PARALLEL_BLOCKS - 1) // NUM_PARALLEL_BLOCKS
        
        # Load ALL blocks' idx and val once at the beginning
        for block_idx in range(num_blocks):
            ps = persistent_scratch[block_idx]
            body.append(("load", ("vload", ps['idx'], idx_ptrs + block_idx)))
            body.append(("load", ("vload", ps['val'], val_ptrs + block_idx)))
        
        for round in range(rounds):
            # Prologue: Compute addresses and scatter load for first group
            first_temps = group_temps[0]
            for local_idx in range(NUM_PARALLEL_BLOCKS):
                block_idx = local_idx
                ps = persistent_scratch[block_idx]
                t = first_temps[local_idx]
                body.append(("valu", ("+", t['addr'], v_forest_p, ps['idx'])))
            for j in range(VLEN):
                for local_idx in range(NUM_PARALLEL_BLOCKS):
                    t = first_temps[local_idx]
                    body.append(("load", ("load_offset", t['node_val'], t['addr'], j)))
            
            # Main loop: process current group while loading next group
            for group_idx in range(num_groups):
                curr_buf = group_idx % 2
                next_buf = 1 - curr_buf
                curr_temps = group_temps[curr_buf]
                next_temps = group_temps[next_buf]
                group_start_block = group_idx * NUM_PARALLEL_BLOCKS
                
                has_next = group_idx + 1 < num_groups
                next_start_block = (group_idx + 1) * NUM_PARALLEL_BLOCKS
                
                # XOR with node_val for current group (val is in persistent_scratch)
                for local_idx in range(NUM_PARALLEL_BLOCKS):
                    block_idx = group_start_block + local_idx
                    ps = persistent_scratch[block_idx]
                    t = curr_temps[local_idx]
                    body.append(("valu", ("^", ps['val'], ps['val'], t['node_val'])))
                
                # Hash computation interleaved with next group's operations
                hash_slots = []
                for stage_idx in range(len(HASH_STAGES)):
                    op1, val1, op2, op3, val3 = HASH_STAGES[stage_idx]
                    for local_idx in range(NUM_PARALLEL_BLOCKS):
                        block_idx = group_start_block + local_idx
                        ps = persistent_scratch[block_idx]
                        t = curr_temps[local_idx]
                        if op1 == "+" and op2 == "+" and op3 == "<<":
                            factor = 1 + (1 << val3)
                            hash_slots.append(("valu", ("multiply_add", ps['val'], ps['val'], self.scratch_vconst(factor), self.scratch_vconst(val1))))
                        else:
                            hash_slots.append(("valu", (op1, t['tmp1'], ps['val'], self.scratch_vconst(val1))))
                            hash_slots.append(("valu", (op3, t['tmp2'], ps['val'], self.scratch_vconst(val3))))
                            hash_slots.append(("valu", (op2, ps['val'], t['tmp1'], t['tmp2'])))
                
                # Interleave hash with next group's address computation and scatter loads
                if has_next:
                    next_addr_slots = []
                    for local_idx in range(NUM_PARALLEL_BLOCKS):
                        next_block_idx = next_start_block + local_idx
                        ps = persistent_scratch[next_block_idx]
                        t = next_temps[local_idx]
                        next_addr_slots.append(("valu", ("+", t['addr'], v_forest_p, ps['idx'])))
                    
                    scatter_slots = []
                    for j in range(VLEN):
                        for local_idx in range(NUM_PARALLEL_BLOCKS):
                            t = next_temps[local_idx]
                            scatter_slots.append(("load", ("load_offset", t['node_val'], t['addr'], j)))
                    
                    h_idx = 0
                    a_idx = 0
                    while h_idx < len(hash_slots) or a_idx < len(next_addr_slots):
                        for _ in range(5):
                            if h_idx < len(hash_slots):
                                body.append(hash_slots[h_idx])
                                h_idx += 1
                        if a_idx < len(next_addr_slots):
                            body.append(next_addr_slots[a_idx])
                            a_idx += 1
                    
                    l_idx = 0
                    while h_idx < len(hash_slots) or l_idx < len(scatter_slots):
                        for _ in range(6):
                            if h_idx < len(hash_slots):
                                body.append(hash_slots[h_idx])
                                h_idx += 1
                        for _ in range(2):
                            if l_idx < len(scatter_slots):
                                body.append(scatter_slots[l_idx])
                                l_idx += 1
                else:
                    body.extend(hash_slots)
                
                # Index computation for current group - interleave across blocks
                # Stage 0: AND
                for local_idx in range(NUM_PARALLEL_BLOCKS):
                    block_idx = group_start_block + local_idx
                    ps = persistent_scratch[block_idx]
                    t = curr_temps[local_idx]
                    body.append(("valu", ("&", t['tmp3'], ps['val'], vone_const)))
                # Stage 1: ADD
                for local_idx in range(NUM_PARALLEL_BLOCKS):
                    t = curr_temps[local_idx]
                    body.append(("valu", ("+", t['tmp3'], t['tmp3'], vone_const)))
                # Stage 2: MULTIPLY_ADD
                for local_idx in range(NUM_PARALLEL_BLOCKS):
                    block_idx = group_start_block + local_idx
                    ps = persistent_scratch[block_idx]
                    t = curr_temps[local_idx]
                    body.append(("valu", ("multiply_add", ps['idx'], ps['idx'], vtwo_const, t['tmp3'])))
                # Stage 3: LESS_THAN
                for local_idx in range(NUM_PARALLEL_BLOCKS):
                    block_idx = group_start_block + local_idx
                    ps = persistent_scratch[block_idx]
                    t = curr_temps[local_idx]
                    body.append(("valu", ("<", t['tmp1'], ps['idx'], v_n_nodes)))
                # Stage 4: MULTIPLY
                for local_idx in range(NUM_PARALLEL_BLOCKS):
                    block_idx = group_start_block + local_idx
                    ps = persistent_scratch[block_idx]
                    t = curr_temps[local_idx]
                    body.append(("valu", ("*", ps['idx'], t['tmp1'], ps['idx'])))

                # Store final results for this group on last round
                if round == rounds - 1:
                    for local_idx in range(NUM_PARALLEL_BLOCKS):
                        block_idx = group_start_block + local_idx
                        ps = persistent_scratch[block_idx]
                        body.append(("store", ("vstore", idx_ptrs + block_idx, ps['idx'])))
                        body.append(("store", ("vstore", val_ptrs + block_idx, ps['val'])))
        
        body_instrs = self.build(body)
        self.instrs.extend(body_instrs)
        for instr in self.instrs:
            with open("kernel_instr_counts.txt", "a") as f:
                for key, val in instr.items():
                    f.write(f"{key}: {len(val)} / {SLOT_LIMITS[key]}, \t")
                f.write("\n")
        # Required to match with the yield in reference_kernel2
        self.instrs.append({"flow": [("pause",)]})

BASELINE = 147734

def do_kernel_test(
    forest_height: int,
    rounds: int,
    batch_size: int,
    seed: int = 123,
    trace: bool = False,
    prints: bool = False,
):
    print(f"{forest_height=}, {rounds=}, {batch_size=}")
    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    kb = KernelBuilder()
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)
    # print(kb.instrs)

    value_trace = {}
    machine = Machine(
        mem,
        kb.instrs,
        kb.debug_info(),
        n_cores=N_CORES,
        value_trace=value_trace,
        trace=trace,
    )
    machine.prints = prints
    for i, ref_mem in enumerate(reference_kernel2(mem, value_trace)):
        machine.run()
        inp_values_p = ref_mem[6]
        if prints:
            print(machine.mem[inp_values_p : inp_values_p + len(inp.values)])
            print(ref_mem[inp_values_p : inp_values_p + len(inp.values)])
        assert (
            machine.mem[inp_values_p : inp_values_p + len(inp.values)]
            == ref_mem[inp_values_p : inp_values_p + len(inp.values)]
        ), f"Incorrect result on round {i}"
        inp_indices_p = ref_mem[5]
        if prints:
            print(machine.mem[inp_indices_p : inp_indices_p + len(inp.indices)])
            print(ref_mem[inp_indices_p : inp_indices_p + len(inp.indices)])
        # Updating these in memory isn't required, but you can enable this check for debugging
        # assert machine.mem[inp_indices_p:inp_indices_p+len(inp.indices)] == ref_mem[inp_indices_p:inp_indices_p+len(inp.indices)]

    print("CYCLES: ", machine.cycle)
    print("Speedup over baseline: ", BASELINE / machine.cycle)
    return machine.cycle


class Tests(unittest.TestCase):
    def test_ref_kernels(self):
        """
        Test the reference kernels against each other
        """
        random.seed(123)
        for i in range(10):
            f = Tree.generate(4)
            inp = Input.generate(f, 10, 6)
            mem = build_mem_image(f, inp)
            reference_kernel(f, inp)
            for _ in reference_kernel2(mem, {}):
                pass
            assert inp.indices == mem[mem[5] : mem[5] + len(inp.indices)]
            assert inp.values == mem[mem[6] : mem[6] + len(inp.values)]

    def test_kernel_trace(self):
        # Full-scale example for performance testing
        do_kernel_test(10, 16, 256, trace=True, prints=False)

    # Passing this test is not required for submission, see submission_tests.py for the actual correctness test
    # You can uncomment this if you think it might help you debug
    # def test_kernel_correctness(self):
    #     for batch in range(1, 3):
    #         for forest_height in range(3):
    #             do_kernel_test(
    #                 forest_height + 2, forest_height + 4, batch * 16 * VLEN * N_CORES
    #             )

    def test_kernel_cycles(self):
        do_kernel_test(10, 16, 256)


# To run all the tests:
#    python perf_takehome.py
# To run a specific test:
#    python perf_takehome.py Tests.test_kernel_cycles
# To view a hot-reloading trace of all the instructions:  **Recommended debug loop**
# NOTE: The trace hot-reloading only works in Chrome. In the worst case if things aren't working, drag trace.json onto https://ui.perfetto.dev/
#    python perf_takehome.py Tests.test_kernel_trace
# Then run `python watch_trace.py` in another tab, it'll open a browser tab, then click "Open Perfetto"
# You can then keep that open and re-run the test to see a new trace.

# To run the proper checks to see which thresholds you pass:
#    python tests/submission_tests.py

if __name__ == "__main__":
    unittest.main()
