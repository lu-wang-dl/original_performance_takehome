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

        def flush_bundle():
            nonlocal bundle, bundle_input, bundle_output
            if bundle:
                instrs.append(bundle)
            bundle = {}
            bundle_input = set()
            bundle_output = set()
        
        def can_add_to_bundle(engine, slot_inputs, slot_outputs):
            if len(bundle.get(engine, [])) >= SLOT_LIMITS[engine]:
                return False
            if slot_inputs & bundle_output:
                return False
            if slot_outputs & bundle_output:
                return False
            return True
        
        idx = 0
        while idx < n:
            if used[idx]:
                idx += 1
                continue
            engine, slot = slots[idx]
            slot_inputs, slot_outputs, barrier = self._slot_input_output(engine, slot)
            if barrier:
                flush_bundle()
                instrs.append({engine: [slot]})
                used[idx] = True
                idx += 1
                continue

            if can_add_to_bundle(engine, slot_inputs, slot_outputs):
                bundle.setdefault(engine, []).append(slot)
                bundle_input.update(slot_inputs)
                bundle_output.update(slot_outputs)
                used[idx] = True
                idx += 1

                # Look ahead: Add slots that do not conflict with the current bundle
                skipped_outputs = set()
                for look_ahead in range(idx, min(n, idx+self.LOOK_AHEAD_NUMBER)):
                    if used[look_ahead]: continue
                    look_ahead_engine, look_ahead_slot = slots[look_ahead]
                    look_ahead_inputs, look_ahead_outputs, look_ahead_barrier = self._slot_input_output(look_ahead_engine, look_ahead_slot)
                    if look_ahead_barrier:
                        break
                    if look_ahead_inputs & skipped_outputs:
                        skipped_outputs.update(look_ahead_inputs)
                        continue
                    # Can't add if slot writes to skipped slot's output (WAW - would overwrite wrong value)
                    if look_ahead_inputs & skipped_outputs:
                        skipped_outputs.update(look_ahead_inputs)
                        continue
                    if can_add_to_bundle(look_ahead_engine, look_ahead_inputs, look_ahead_outputs):
                        bundle.setdefault(look_ahead_engine, []).append(look_ahead_slot)
                        bundle_input.update(look_ahead_inputs)
                        bundle_output.update(look_ahead_outputs)
                        used[look_ahead] = True
                    else:
                        skipped_outputs.update(look_ahead_outputs)
            else:
                flush_bundle()
                bundle.setdefault(engine, []).append(slot)
                bundle_input.update(slot_inputs)
                bundle_output.update(slot_outputs)
                used[idx] = True
                idx += 1

        flush_bundle()
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
        Vectorized implementation using SIMD for batch dimension.
        Assume batch_size is a multiple of VLEN.
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
        
        # Vector scratch registers and broadcasted constants.
        vtmp1 = self.alloc_scratch("vtmp1", VLEN)
        vtmp2 = self.alloc_scratch("vtmp2", VLEN)
        vtmp3 = self.alloc_scratch("vtmp3", VLEN)

        vone_const = self.scratch_vconst(1, "vone")
        vtwo_const = self.scratch_vconst(2, "vtwo")

        vtmp_idx = self.alloc_scratch("vtmp_idx", VLEN)
        vtmp_val = self.alloc_scratch("vtmp_val", VLEN)
        vtmp_node_val = self.alloc_scratch("vtmp_node_val", VLEN)
        vtmp_addr = self.alloc_scratch("vtmp_addr", VLEN)
        
        v_n_nodes = self.alloc_scratch("v_n_nodes", VLEN)
        self.add("valu", ("vbroadcast", v_n_nodes, self.scratch["n_nodes"]))
        v_forest_p = self.alloc_scratch("v_forest_p", VLEN)
        self.add("valu", ("vbroadcast", v_forest_p, self.scratch["forest_values_p"]))

        # Pause instructions are matched up with yield statements in the reference
        # kernel to let you debug at intermediate steps. The testing harness in this
        # file requires these match up to the reference kernel's yields, but the
        # submission harness ignores them.
        self.add("flow", ("pause",))
        if self.enable_debug:
            self.add("debug", ("comment", "Starting loop"))

        body = []  # array of slots

        # Scalar scratch registers
        tmp_idx = self.alloc_scratch("tmp_idx")
        tmp_val = self.alloc_scratch("tmp_val")
        tmp_node_val = self.alloc_scratch("tmp_node_val")
        tmp_addr = self.alloc_scratch("tmp_addr")

        for round in range(rounds):
            for i in range(0, batch_size, VLEN):
                # idx = mem[inp_indices_p + i]
                # val = mem[inp_values_p + i]
                idx_ptr = idx_ptrs + i
                val_ptr = val_ptrs + i
                body.append(("load", ("vload", vtmp_idx, idx_ptr)))                
                body.append(("load", ("vload", vtmp_val, val_ptr)))

                if self.enable_debug:
                    body.append(("debug", ("vcompare", vtmp_idx, [(round, i+j, "idx") for j in range(VLEN)])))
                    body.append(("debug", ("vcompare", vtmp_val, [(round, i+j, "val") for j in range(VLEN)])))
                # node_val = mem[forest_values_p + idx]
                body.append(("valu", ("+", vtmp_addr, v_forest_p, vtmp_idx)))
                for j in range(VLEN):
                    body.append(("load", ("load_offset", vtmp_node_val, vtmp_addr, j)))
                if self.enable_debug:
                    body.append(("debug", ("vcompare", vtmp_node_val, [(round, i+j, "node_val") for j in range(VLEN)])))
                # val = myhash(val ^ node_val)
                body.append(("valu", ("^", vtmp_val, vtmp_val, vtmp_node_val)))
                body.extend(self.build_vhash(vtmp_val, vtmp1, vtmp2, round, i))
                if self.enable_debug:
                    body.append(("debug", ("vcompare", vtmp_val, [(round, i+j, "hashed_val") for j in range(VLEN)])))
                # Change to idx = 2*idx + 1 + (val % 2)
                body.append(("valu", ("&", vtmp3, vtmp_val, vone_const)))
                body.append(("valu", ("+", vtmp3, vtmp3, vone_const)))
                body.append(("valu", ("multiply_add", vtmp_idx, vtmp_idx, vtwo_const, vtmp3)))
                if self.enable_debug:
                    body.append(("debug", ("vcompare", vtmp_idx, [(round, i+j, "next_idx") for j in range(VLEN)])))
                # Change to idx = idx * (idx < n_nodes)
                body.append(("valu", ("<", vtmp1, vtmp_idx, v_n_nodes)))
                body.append(("valu", ("*", vtmp_idx, vtmp1, vtmp_idx)))
                if self.enable_debug:
                    body.append(("debug", ("vcompare", vtmp_idx, [(round, i+j, "wrapped_idx") for j in range(VLEN)])))
                # mem[inp_indices_p + i] = idx
                body.append(("store", ("vstore", idx_ptr, vtmp_idx)))
                # mem[inp_values_p + i] = val
                body.append(("store", ("vstore", val_ptr, vtmp_val)))

        body_instrs = self.build(body)
        self.instrs.extend(body_instrs)
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
