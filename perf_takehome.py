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

    LOOK_AHEAD_NUMBER = 100
    NUM_PARALLEL_BLOCKS = 8

    def __init__(self):
        self.instrs = []
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.const_map = {}
        self.vconst_map = {}
        self.enable_debug = False
        self.enable_bundle_logging = False

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)
    
    def _slot_input_output(self, engine, slot):
        inputs = set()
        outputs = set()
        barrier = False

        def add_vec(base):
            return set(range(base, base + VLEN))

        if engine == "debug":
            return inputs, outputs, True

        if engine == "alu":
            _, dest, a1, a2 = slot
            inputs.update([a1, a2])
            outputs.add(dest)
        elif engine == "valu":
            op = slot[0]
            if op == "vbroadcast":
                _, dest, src = slot
                inputs.add(src)
                outputs.update(add_vec(dest))
            elif op == "multiply_add":
                _, dest, a, b, c = slot
                inputs.update(add_vec(a))
                inputs.update(add_vec(b))
                inputs.update(add_vec(c))
                outputs.update(add_vec(dest))
            else:
                _, dest, a1, a2 = slot
                inputs.update(add_vec(a1))
                inputs.update(add_vec(a2))
                outputs.update(add_vec(dest))
        elif engine == "load":
            match slot:
                case ("load", dest, addr):
                    inputs.add(addr)
                    outputs.add(dest)
                case ("load_offset", dest, addr, offset):
                    inputs.add(addr + offset)
                    outputs.add(dest + offset)
                case ("vload", dest, addr):
                    inputs.add(addr)
                    outputs.update(add_vec(dest))
                case ("const", dest, _val):
                    outputs.add(dest)
        elif engine == "store":
            match slot:
                case ("store", addr, src):
                    inputs.update([addr, src])
                case ("vstore", addr, src):
                    inputs.add(addr)
                    inputs.update(add_vec(src))
        elif engine == "flow":
            op = slot[0]
            if op in ("halt", "pause", "jump", "jump_indirect", "cond_jump", "cond_jump_rel"):
                barrier = True
            match slot:
                case ("select", dest, cond, a, b):
                    inputs.update([cond, a, b])
                    outputs.add(dest)
                case ("add_imm", dest, a, _imm):
                    inputs.add(a)
                    outputs.add(dest)
                case ("vselect", dest, cond, a, b):
                    inputs.update(add_vec(cond))
                    inputs.update(add_vec(a))
                    inputs.update(add_vec(b))
                    outputs.update(add_vec(dest))
                case ("trace_write", val):
                    inputs.add(val)
                case ("coreid", dest):
                    outputs.add(dest)

        return inputs, outputs, barrier

    def build(self, slots: list[tuple[Engine, tuple]], vliw: bool = False):
        # Update the logic for slot packing
        instrs = []
        used = [False] * len(slots)
        n = len(slots)

        bundle = {}
        bundle_input = set()
        bundle_output = set()

        # Add a flag to enable/disable logging
        enable_logging = getattr(self, "enable_bundle_logging", False)

        import os
        import inspect

        log_file_path = "bundle_decisions.log"
        if enable_logging:
            # Optionally, clear the log before starting (can comment out if desired)
            with open(log_file_path, "w") as log_file:
                log_file.write("")  # Clear file before logging

        def log_decision(message):
            if not enable_logging:
                return
            frame = inspect.currentframe().f_back
            lineno = frame.f_lineno
            with open(log_file_path, "a") as log_file:
                log_file.write(f"[line {lineno}] {message}\n")

        def flush_bundle():
            nonlocal bundle, bundle_input, bundle_output
            if bundle:
                instrs.append(bundle)
            log_decision(f"Flush: {bundle}, bundle_input: {bundle_input}, bundle_output: {bundle_output}")
            bundle = {}
            bundle_input = set()
            bundle_output = set()
        
        def can_add_to_bundle(engine, slot_inputs, slot_outputs):
            reason = None
            if len(bundle.get(engine, [])) >= SLOT_LIMITS[engine]:
                reason = f"SLOT_LIMIT: engine={engine} has {len(bundle.get(engine, []))} (limit {SLOT_LIMITS[engine]})"
                log_decision(f"can_add_to_bundle: NO (engine={engine}) - {reason}")
                return False
            if slot_inputs & bundle_output:
                reason = f"RAW hazard: slot_inputs {slot_inputs} & bundle_output {bundle_output} = {slot_inputs & bundle_output}"
                log_decision(f"can_add_to_bundle: NO (engine={engine}) - {reason}")
                return False
            if slot_outputs & bundle_output:
                reason = f"WAW hazard: slot_outputs {slot_outputs} & bundle_output {bundle_output} = {slot_outputs & bundle_output}"
                log_decision(f"can_add_to_bundle: NO (engine={engine}) - {reason}")
                return False
            log_decision(f"can_add_to_bundle: YES (engine={engine})")
            return True
        
        idx = 0
        while idx < n:
            if used[idx]:
                idx += 1
                continue
            engine, slot = slots[idx]
            slot_inputs, slot_outputs, barrier = self._slot_input_output(engine, slot)
            log_decision(f"Step idx={idx}, engine={engine}, slot={slot}, barrier={barrier}")
            if barrier:
                flush_bundle()
                instrs.append({engine: [slot]})
                log_decision(f"Barrier at idx={idx}: {engine}, slot={slot} placed in new instr; advancing.")
                used[idx] = True
                idx += 1
                continue

            if can_add_to_bundle(engine, slot_inputs, slot_outputs):
                log_decision(f"At idx={idx}, can add: engine={engine}, slot={slot} to current bundle.")
                bundle.setdefault(engine, []).append(slot)
                bundle_input.update(slot_inputs)
                bundle_output.update(slot_outputs)
                used[idx] = True
                idx += 1
            else:
                log_decision(f"At idx={idx}, cannot add: engine={engine}, slot={slot}. Flushing bundle and starting new bundle with this slot.")
                flush_bundle()
                bundle.setdefault(engine, []).append(slot)
                bundle_input.update(slot_inputs)
                bundle_output.update(slot_outputs)
                used[idx] = True
                idx += 1

            # Look ahead: Add slots that do not conflict with the current bundle
            skipped_outputs = set()
            for look_ahead in range(idx, min(n, idx+self.LOOK_AHEAD_NUMBER)):
                if used[look_ahead]:
                    log_decision(f"  Lookahead {look_ahead}: already used, skipping.")
                    continue
                look_ahead_engine, look_ahead_slot = slots[look_ahead]
                look_ahead_inputs, look_ahead_outputs, look_ahead_barrier = self._slot_input_output(look_ahead_engine, look_ahead_slot)
                log_decision(f"  Lookahead {look_ahead}: engine={look_ahead_engine}, slot={look_ahead_slot}")
                if look_ahead_barrier:
                    log_decision(f"  Lookahead {look_ahead}: barrier encountered, breaking lookahead.")
                    break
                # RAW: Can't add if slot reads from a skipped slot's output
                if look_ahead_inputs & skipped_outputs:
                    log_decision(f"  Lookahead {look_ahead}: RAW hazard (inputs {look_ahead_inputs} & skipped_outputs {skipped_outputs}), skipping.")
                    skipped_outputs.update(look_ahead_outputs)
                    continue
                # WAW: Can't add if slot writes to a skipped slot's output
                if look_ahead_outputs & skipped_outputs:
                    log_decision(f"  Lookahead {look_ahead}: WAW hazard (outputs {look_ahead_outputs} & skipped_outputs {skipped_outputs}), skipping.")
                    skipped_outputs.update(look_ahead_outputs)
                    continue
                if can_add_to_bundle(look_ahead_engine, look_ahead_inputs, look_ahead_outputs):
                    log_decision(f"  Lookahead {look_ahead}: can add to bundle: engine={look_ahead_engine}, slot={look_ahead_slot}.")
                    bundle.setdefault(look_ahead_engine, []).append(look_ahead_slot)
                    bundle_input.update(look_ahead_inputs)
                    bundle_output.update(look_ahead_outputs)
                    used[look_ahead] = True
                else:
                    log_decision(f"  Lookahead {look_ahead}: cannot add to bundle, updating skipped_outputs (outputs {look_ahead_outputs}).")
                    skipped_outputs.update(look_ahead_outputs)

        flush_bundle()
        log_decision("Build finished\n")
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

    def scratch_const(self, val, name=None):
        if val not in self.const_map:
            addr = self.alloc_scratch(name)
            self.add("load", ("const", addr, val))
            self.const_map[val] = addr
        return self.const_map[val]
    
    def scratch_vconst(self, val, name=None):
        if val not in self.vconst_map:
            addr = self.alloc_scratch(name, length=VLEN)
            scalar_addr = self.scratch_const(val)
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
        Vectorized implementation using SIMD for batch dimension with software pipelining.
        Assume batch_size is a multiple of VLEN.
        
        Pipeline structure:
        - Stage 0: Load idx and val from memory
        - Stage 1: Compute addr, start scattered node_val loads (first 2)
        - Stage 2: Continue scattered node_val loads (next 2)
        - Stage 3: Continue scattered node_val loads (next 2)
        - Stage 4: Finish scattered node_val loads (last 2), start hash
        - Stage 5: Continue hash computation + index update + stores
        
        By interleaving iterations at different pipeline stages, we overlap
        the load bottleneck with computation.
        """

        zero_const = self.scratch_const(0)
        one_const = self.scratch_const(1)

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
            self.alloc_scratch(v)

        header_base = self.alloc_scratch("header", VLEN)
        self.add("load", ("vload", header_base, zero_const))
        # Map init_vars to their positions in the header block
        for i, v in enumerate(init_vars):
            self.scratch[v] = header_base + i
            self.scratch_debug[header_base + i] = (v, 1)

        # Precompute the idx: inp_indices_p + i and inp_values_p + i
        idx_ptrs = self.alloc_scratch("idx_ptrs", batch_size)
        val_ptrs = self.alloc_scratch("val_ptrs", batch_size)
        self.add("alu", ("+", idx_ptrs, self.scratch["inp_indices_p"], zero_const))
        self.add("alu", ("+", val_ptrs, self.scratch["inp_values_p"], zero_const))
        for i in range(1, batch_size):
            self.add("alu", ("+", idx_ptrs + i, idx_ptrs + i - 1, one_const))
            self.add("alu", ("+", val_ptrs + i, val_ptrs + i - 1, one_const))   

        vone_const = self.scratch_vconst(1, "vone")
        vtwo_const = self.scratch_vconst(2, "vtwo")
        v_n_nodes = self.alloc_scratch("v_n_nodes", VLEN)
        self.add("valu", ("vbroadcast", v_n_nodes, self.scratch["n_nodes"]))
        v_forest_p = self.alloc_scratch("v_forest_p", VLEN)
        self.add("valu", ("vbroadcast", v_forest_p, self.scratch["forest_values_p"]))

        scratch_blocks = []
        for i in range(self.NUM_PARALLEL_BLOCKS):
            scratch_blocks.append({
                "idx": self.alloc_scratch(f"block_{i}_idx", VLEN),
                "val": self.alloc_scratch(f"block_{i}_val", VLEN),
                "node_val": self.alloc_scratch(f"block_{i}_node_val", VLEN),
                "addr": self.alloc_scratch(f"block_{i}_addr", VLEN),
                "vtmp1": self.alloc_scratch(f"block_{i}_vtmp1", VLEN),
                "vtmp2": self.alloc_scratch(f"block_{i}_vtmp2", VLEN),
                "vtmp3": self.alloc_scratch(f"block_{i}_vtmp3", VLEN),
            })

        self.add("flow", ("pause",))
        if self.enable_debug:
            self.add("debug", ("comment", "Starting loop"))

        body = []  # array of slots

        # Total number of vector iterations
        n_iters = rounds * (batch_size // VLEN)
        
        def get_iter_params(iter_idx):
            """Get round and batch offset for a given iteration index."""
            if iter_idx < 0 or iter_idx >= n_iters:
                return None
            iters_per_round = batch_size // VLEN
            round_num = iter_idx // iters_per_round
            batch_offset = (iter_idx % iters_per_round) * VLEN
            return (round_num, batch_offset)
        
        def get_block(iter_idx):
            """Get scratch block for an iteration."""
            return scratch_blocks[iter_idx % self.NUM_PARALLEL_BLOCKS]
        
        def emit_stage0_load_idx_val(iter_idx):
            """Stage 0: Load idx and val vectors from memory."""
            params = get_iter_params(iter_idx)
            if params is None:
                return []
            round_num, i = params
            block = get_block(iter_idx)
            vtmp_idx = block["idx"]
            vtmp_val = block["val"]

            slots = []

            # idx = mem[inp_indices_p + i]
            # val = mem[inp_values_p + i]
            idx_ptr = idx_ptrs + i
            val_ptr = val_ptrs + i
            slots.append(("load", ("vload", vtmp_idx, idx_ptr)))
            slots.append(("load", ("vload", vtmp_val, val_ptr)))
            if self.enable_debug:
                slots.append(("debug", ("vcompare", vtmp_idx, [(round_num, i+j, "idx") for j in range(VLEN)])))
                slots.append(("debug", ("vcompare", vtmp_val, [(round_num, i+j, "val") for j in range(VLEN)])))
            return slots
        
        def emit_stage1_loads_node_val(iter_idx):
            """Stage 1: loads tree node values."""
            params = get_iter_params(iter_idx)
            if params is None:
                return []
            round_num, i = params
            block = get_block(iter_idx)
            vtmp_idx = block["idx"]
            vtmp_addr = block["addr"]
            vtmp_node_val = block["node_val"]
            
            slots = []
            # node_val = mem[forest_values_p + idx]
            # Compute address first
            slots.append(("valu", ("+", vtmp_addr, v_forest_p, vtmp_idx)))
            # Emit loads in pairs to maximize bundling with load engine (2 slots/cycle limit)
            # This allows the bundler to pack 2 loads per cycle more efficiently
            for j in range(VLEN):
                slots.append(("load", ("load_offset", vtmp_node_val, vtmp_addr, j)))
            if self.enable_debug:
                slots.append(("debug", ("vcompare", vtmp_node_val, [(round_num, i+j, "node_val") for j in range(VLEN)])))
            return slots
        
        def emit_stage2_xor(iter_idx):
            """Stage 2: XOR."""
            params = get_iter_params(iter_idx)
            if params is None:
                return []
            block = get_block(iter_idx)
            vtmp_val = block["val"]
            vtmp_node_val = block["node_val"]
            
            slots = []
            # val = val ^ node_val
            slots.append(("valu", ("^", vtmp_val, vtmp_val, vtmp_node_val)))
            return slots
        
        def emit_hash(iter_idx):
            """Hash stages."""
            params = get_iter_params(iter_idx)
            if params is None:
                return []
            round_num, i = params
            block = get_block(iter_idx)
            slots = []
            # # val = myhash(val)
            for hi in range(6):
                op1, val1, op2, op3, val3 = HASH_STAGES[hi]
                if op1 == "+" and op2 == "+" and op3 == "<<":
                    factor = 1 + (1 << val3)
                    slots.append(("valu", ("multiply_add", block["val"], block["val"], 
                                          self.scratch_vconst(factor), self.scratch_vconst(val1))))
                else:
                    slots.append(("valu", (op1, block["vtmp1"], block["val"], self.scratch_vconst(val1))))
                    slots.append(("valu", (op3, block["vtmp2"], block["val"], self.scratch_vconst(val3))))
                    slots.append(("valu", (op2, block["val"], block["vtmp1"], block["vtmp2"])))
            if self.enable_debug:
                slots.append(("debug", ("vcompare", block["val"], [(round_num, i+j, "hashed_val") for j in range(VLEN)])))
            return slots
        
        def emit_idx_update(iter_idx):
            """Index update operations."""
            params = get_iter_params(iter_idx)
            if params is None:
                return []
            round_num, i = params
            block = get_block(iter_idx)
            slots = []
            # idx = 2*idx + 1 + (val & 1)
            slots.append(("valu", ("&", block["vtmp3"], block["val"], vone_const)))
            slots.append(("valu", ("+", block["vtmp3"], block["vtmp3"], vone_const)))
            slots.append(("valu", ("multiply_add", block["idx"], block["idx"], vtwo_const, block["vtmp3"])))
            if self.enable_debug:
                slots.append(("debug", ("vcompare", block["idx"], [(round_num, i+j, "next_idx") for j in range(VLEN)])))
            # idx = 0 if idx >= n_nodes else idx
            # idx = idx * (idx < n_nodes)
            slots.append(("valu", ("<", block["vtmp1"], block["idx"], v_n_nodes)))
            slots.append(("valu", ("*", block["idx"], block["vtmp1"], block["idx"])))
            if self.enable_debug:
                slots.append(("debug", ("vcompare", block["idx"], [(round_num, i+j, "wrapped_idx") for j in range(VLEN)])))
            return slots
        
        def emit_stores(iter_idx):
            """Store operations."""
            params = get_iter_params(iter_idx)
            if params is None:
                return []
            round_num, i = params
            block = get_block(iter_idx)
            idx_ptr = idx_ptrs + i
            val_ptr = val_ptrs + i
            slots = []
            # mem[inp_indices_p + i] = idx
            # mem[inp_values_p + i] = val
            slots.append(("store", ("vstore", idx_ptr, block["idx"])))
            slots.append(("store", ("vstore", val_ptr, block["val"])))
            return slots
        
        # Software pipelined execution with 6 stages (global across all rounds)
        n_iters = rounds * (batch_size // VLEN)
        NUM_STAGES = 6
        total_steps = n_iters + NUM_STAGES - 1
        
        for step in range(total_steps):
            # Reorder stages to maximize bundling opportunities
            # Emit operations in an order that allows better interleaving:
            # 1. First emit VALU operations (hash, idx update) which have more slots available
            # 2. Then emit loads which are more constrained
            # 3. This allows the bundler to pack VALU ops with loads from different iterations
            
            # Stage 3: Hash (VALU operations - 6 slots available)
            iter_s3 = step - 3
            if 0 <= iter_s3 < n_iters:
                body.extend(emit_hash(iter_s3))
            
            # Stage 4: Idx update (VALU operations - 6 slots available)
            iter_s4 = step - 4
            if 0 <= iter_s4 < n_iters:
                body.extend(emit_idx_update(iter_s4))
            
            # Stage 2: XOR (VALU operation - can be bundled with hash/idx_update)
            iter_s2 = step - 2
            if 0 <= iter_s2 < n_iters:
                body.extend(emit_stage2_xor(iter_s2))
            
            # Stage 0: Load idx/val (load operations - 2 slots available)
            if step < n_iters:
                body.extend(emit_stage0_load_idx_val(step))
            
            # Stage 1: Load node val (load operations - 2 slots available, most constrained)
            iter_s1 = step - 1
            if 0 <= iter_s1 < n_iters:
                body.extend(emit_stage1_loads_node_val(iter_s1))
            
            # Stage 5: Stores (store operations - 2 slots available)
            iter_s5 = step - 5
            if 0 <= iter_s5 < n_iters:
                body.extend(emit_stores(iter_s5))

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
        do_kernel_test(10, 1, 16)


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
