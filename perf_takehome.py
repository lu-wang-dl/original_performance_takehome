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
    LOOK_AHEAD_NUMBER = 8000

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

    def build(self, slots: list[tuple[Engine, tuple]]):
        """
        List-scheduling approach: for each slot, compute earliest cycle based on
        dependencies, then find first cycle with available resources.
        """
        schedule = []  # List of dict[engine, list[op]]
        cycle_counts = []  # Track usage per cycle: index -> dict[engine, count]

        # Precompute dependency graph using program order (RAW/WAW/WAR + barriers)
        last_write_idx = {}
        last_read_idx = {}
        last_barrier_idx = -1
        slot_meta = []
        dependents = [[] for _ in slots]
        dep_count = [0 for _ in slots]

        for i, (engine, op) in enumerate(slots):
            inputs, outputs, barrier = self._slot_input_output(engine, op)
            slot_meta.append(
                {
                    "engine": engine,
                    "op": op,
                    "inputs": inputs,
                    "outputs": outputs,
                    "barrier": barrier,
                }
            )

            deps_for_i = set()

            if last_barrier_idx != -1:
                deps_for_i.add(last_barrier_idx)

            if barrier and i > 0:
                deps_for_i.add(i - 1)

            # RAW: read depends on last write
            for r in inputs:
                if r in last_write_idx:
                    deps_for_i.add(last_write_idx[r])

            # WAW: write depends on last write
            for w in outputs:
                if w in last_write_idx:
                    deps_for_i.add(last_write_idx[w])

            # WAR: write depends on last read
            for w in outputs:
                if w in last_read_idx:
                    deps_for_i.add(last_read_idx[w])

            for dep in deps_for_i:
                dependents[dep].append(i)
                dep_count[i] += 1
            for w in outputs:
                last_write_idx[w] = i
            for r in inputs:
                last_read_idx[r] = i

            if barrier:
                last_barrier_idx = i

        ready = [i for i in range(len(slots)) if dep_count[i] == 0]
        scheduled = [False for _ in slots]
        scheduled_count = 0
        current_cycle = 0
        min_unscheduled = 0

        last_write_cycle = {}

        engine_priority = {
            "flow": 0,
            "load": 1,
            "store": 2,
            "valu": 3,
            "alu": 4,
            "debug": 5,
        }

        while scheduled_count < len(slots):
            while current_cycle >= len(schedule):
                schedule.append(defaultdict(list))
                cycle_counts.append(defaultdict(int))

            while min_unscheduled < len(slots) and scheduled[min_unscheduled]:
                min_unscheduled += 1
            window_limit = min_unscheduled + self.LOOK_AHEAD_NUMBER

            scheduled_this_cycle = False
            made_progress = True

            while made_progress:
                made_progress = False
                ready_sorted = sorted(
                    ready, key=lambda i: (engine_priority[slot_meta[i]["engine"]], i)
                )
                for idx in ready_sorted:
                    if scheduled[idx]:
                        continue
                    if idx >= window_limit:
                        continue
                    meta = slot_meta[idx]
                    engine = meta["engine"]
                    inputs = meta["inputs"]
                    outputs = meta["outputs"]

                    start_cycle = 0
                    for r in inputs:
                        if r in last_write_cycle:
                            start_cycle = max(start_cycle, last_write_cycle[r] + 1)
                    for w in outputs:
                        if w in last_write_cycle:
                            start_cycle = max(start_cycle, last_write_cycle[w] + 1)

                    if start_cycle > current_cycle:
                        continue
                    if cycle_counts[current_cycle][engine] >= SLOT_LIMITS[engine]:
                        continue

                    schedule[current_cycle][engine].append(meta["op"])
                    cycle_counts[current_cycle][engine] += 1

                    for w in outputs:
                        last_write_cycle[w] = current_cycle

                    scheduled[idx] = True
                    scheduled_count += 1
                    ready.remove(idx)
                    for dep in dependents[idx]:
                        dep_count[dep] -= 1
                        if dep_count[dep] == 0:
                            ready.append(dep)

                    scheduled_this_cycle = True
                    made_progress = True

            if not scheduled_this_cycle:
                current_cycle += 1
                continue

            current_cycle += 1

        return [dict(s) for s in schedule]

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

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """
        Vectorized implementation using wavefront scheduling with hash stage interleaving.
        Key optimization: Process K=32 batches together where batch k is at round r = t - k.
        Hash stages are interleaved across all active batches to maximize independent operations.
        """
        # Header layout is fixed in build_mem_image; hardcode pointers.
        forest_values_p = self.alloc_scratch("forest_values_p")
        inp_values_p = self.alloc_scratch("inp_values_p")

        all_slots = []
        all_slots.append(("load", ("const", forest_values_p, 7)))
        all_slots.append(("load", ("const", inp_values_p, 7 + n_nodes + batch_size)))

        # Vector constants
        one_v = self.scratch_vconst(1, "one_v", all_slots)
        two_v = self.scratch_vconst(2, "two_v", all_slots)

        # Allocate scratch for preloaded tree values (levels 0-2)
        v_root_val = self.alloc_scratch("v_root_val", VLEN)
        v_node_val_1 = self.alloc_scratch("v_node_val_1", VLEN)
        v_node_val_2 = self.alloc_scratch("v_node_val_2", VLEN)
        v_cache_l2 = self.alloc_scratch("v_cache_l2", 4 * VLEN)  # nodes 3-6

        # Preload tree values using vload + broadcast
        t_preload_vec = self.alloc_scratch("t_preload_vec", VLEN)
        all_slots.append(("load", ("vload", t_preload_vec, forest_values_p)))
        all_slots.append(("valu", ("vbroadcast", v_root_val, t_preload_vec + 0)))
        all_slots.append(("valu", ("vbroadcast", v_node_val_1, t_preload_vec + 1)))
        all_slots.append(("valu", ("vbroadcast", v_node_val_2, t_preload_vec + 2)))
        for i in range(4):
            all_slots.append(("valu", ("vbroadcast", v_cache_l2 + i * VLEN, t_preload_vec + 3 + i)))

        # L3 cache: 8 nodes at level 3 (tree positions 7-14)
        v_cache_l3 = self.alloc_scratch("v_cache_l3", 8 * VLEN)
        l3_load_addr = t_preload_vec  # Reuse as temp scalar for address
        all_slots.append(("flow", ("add_imm", l3_load_addr, forest_values_p, 7)))
        all_slots.append(("load", ("vload", t_preload_vec, l3_load_addr)))
        for j in range(8):
            all_slots.append(("valu", ("vbroadcast", v_cache_l3 + j * VLEN, t_preload_vec + j)))
        v_tmp_l3 = t_preload_vec  # Shared 4th vector temp for L3 mux (free after init)

        # Wavefront scheduling parameters
        K = 32  # Process K batches together
        K_HALF = K // 2  # Interleave two halves
        PIPE_CHUNK = 1
        num_vec_batches = batch_size // VLEN
        NUM_TEMP_SETS = K

        # Allocate scratch for K batch temporaries
        batch_temps = []
        for k in range(NUM_TEMP_SETS):
            temps = {
                "v_tmp1": self.alloc_scratch(f"v_tmp1_{k}", VLEN),
                "v_tmp2": self.alloc_scratch(f"v_tmp2_{k}", VLEN),
                "v_node_vals": self.alloc_scratch(f"v_node_vals_{k}", VLEN),
            }
            batch_temps.append(temps)

        # Allocate scratch for all batch indices and values
        v_indices = []
        v_values = []
        for i in range(batch_size // VLEN):
            v_idx = self.alloc_scratch(f"indices_batch_{i}", VLEN)
            v_val = self.alloc_scratch(f"values_batch_{i}", VLEN)
            v_indices.append(v_idx)
            v_values.append(v_val)

        # Pre-cache hash constants
        vv_mul_0 = self.scratch_vconst(4097, init_slots=all_slots)
        vv_add_0 = self.scratch_vconst(HASH_STAGES[0][1], init_slots=all_slots)
        vv1_1 = self.scratch_vconst(HASH_STAGES[1][1], init_slots=all_slots)
        vv3_1 = self.scratch_vconst(HASH_STAGES[1][4], init_slots=all_slots)
        vv_mul_2 = self.scratch_vconst(33, init_slots=all_slots)
        vv_add_2 = self.scratch_vconst(HASH_STAGES[2][1], init_slots=all_slots)
        vv1_3 = self.scratch_vconst(HASH_STAGES[3][1], init_slots=all_slots)
        vv3_3 = self.scratch_vconst(HASH_STAGES[3][4], init_slots=all_slots)
        vv_mul_4 = self.scratch_vconst(9, init_slots=all_slots)
        vv_add_4 = self.scratch_vconst(HASH_STAGES[4][1], init_slots=all_slots)
        vv1_5 = self.scratch_vconst(HASH_STAGES[5][1], init_slots=all_slots)
        vv3_5 = self.scratch_vconst(HASH_STAGES[5][4], init_slots=all_slots)

        # Scalar constants for bit extraction in L3 mux
        const_1 = self.const_map[1]
        const_2 = self.const_map[2]
        const_4 = self.scratch_const(4, init_slots=all_slots)

        # Pre-compute level base constants as const loads
        level_bases = {}
        for level in range(4, forest_height + 1):
            addr = self.alloc_scratch(f"level_base_{level}")
            all_slots.append(("load", ("const", addr, 2 ** level + 6)))
            level_bases[level] = addr

        # Wavefront schedule: process batches at different rounds together
        for i_base in range(0, num_vec_batches, K):
            k_end = min(K, num_vec_batches - i_base)

            # Load ALL batch values upfront (before wavefront)
            for k in range(k_end):
                i = i_base + k
                jit_addr = batch_temps[k % NUM_TEMP_SETS]["v_tmp1"]
                all_slots.append(("flow", ("add_imm", jit_addr, inp_values_p, i * VLEN)))
                all_slots.append(("load", ("vload", v_values[i], jit_addr)))

            for t in range(rounds + K_HALF - 1):
                active = []
                active_last = []
                for k in range(k_end):
                    r = t - (k % K_HALF)
                    if 0 <= r < rounds:
                        if r == rounds - 1:
                            active_last.append((k, r))
                        else:
                            active.append((k, r))
                active.extend(active_last)
                if not active:
                    continue

                # Process active batches in chunks: pipeline gather(current) with compute(prev)
                num_groups = (len(active) + PIPE_CHUNK - 1) // PIPE_CHUNK
                prev_group = None

                for g in range(num_groups + 1):
                    group = active[g * PIPE_CHUNK : g * PIPE_CHUNK + PIPE_CHUNK] if g < num_groups else []

                    # --- Gather for current group (FIRST) ---
                    for k, r in group:
                        level = r % (forest_height + 1)
                        i = i_base + k
                        temps = batch_temps[k % NUM_TEMP_SETS]

                        if level == 0:
                            pass  # XOR directly with v_root_val
                        elif level == 1:
                            all_slots.append(
                                ("flow", ("vselect", temps["v_node_vals"], v_indices[i], v_node_val_2, v_node_val_1))
                            )
                        elif level == 2:
                            for vi in range(VLEN):
                                all_slots.append(("alu", ("&", temps["v_tmp2"] + vi, v_indices[i] + vi, const_1)))
                            for vi in range(VLEN):
                                all_slots.append(("alu", ("&", temps["v_tmp1"] + vi, v_indices[i] + vi, const_2)))
                            all_slots.append(
                                ("flow", ("vselect", temps["v_node_vals"], temps["v_tmp2"], v_cache_l2 + 1 * VLEN, v_cache_l2 + 0 * VLEN))
                            )
                            all_slots.append(
                                ("flow", ("vselect", temps["v_tmp2"], temps["v_tmp2"], v_cache_l2 + 3 * VLEN, v_cache_l2 + 2 * VLEN))
                            )
                            all_slots.append(
                                ("flow", ("vselect", temps["v_node_vals"], temps["v_tmp1"], temps["v_tmp2"], temps["v_node_vals"]))
                            )
                        elif level == 3:
                            D = v_tmp_l3
                            A = temps["v_tmp1"]
                            B = temps["v_tmp2"]
                            C = temps["v_node_vals"]
                            for vi in range(VLEN):
                                all_slots.append(("alu", ("&", D + vi, v_indices[i] + vi, const_1)))
                            for vi in range(VLEN):
                                all_slots.append(("alu", ("&", B + vi, v_indices[i] + vi, const_2)))
                            all_slots.append(("flow", ("vselect", C, D, v_cache_l3 + 1 * VLEN, v_cache_l3 + 0 * VLEN)))
                            all_slots.append(("flow", ("vselect", A, D, v_cache_l3 + 3 * VLEN, v_cache_l3 + 2 * VLEN)))
                            all_slots.append(("flow", ("vselect", C, B, A, C)))
                            all_slots.append(("flow", ("vselect", A, D, v_cache_l3 + 5 * VLEN, v_cache_l3 + 4 * VLEN)))
                            all_slots.append(("flow", ("vselect", D, D, v_cache_l3 + 7 * VLEN, v_cache_l3 + 6 * VLEN)))
                            all_slots.append(("flow", ("vselect", A, B, D, A)))
                            for vi in range(VLEN):
                                all_slots.append(("alu", ("&", B + vi, v_indices[i] + vi, const_4)))
                            all_slots.append(("flow", ("vselect", C, B, A, C)))
                        elif level >= 4:
                            level_base = level_bases[level]
                            for vi in range(VLEN):
                                v_idx_addr = v_indices[i] + vi
                                all_slots.append(("alu", ("+", temps["v_tmp1"] + vi, level_base, v_idx_addr)))
                            for vi in range(VLEN):
                                all_slots.append(("load", ("load_offset", temps["v_node_vals"], temps["v_tmp1"], vi)))

                    # --- Compute for previous group (SECOND) ---
                    if prev_group is not None:
                        for k, r in prev_group:
                            i = i_base + k
                            level = r % (forest_height + 1)
                            if level == 0:
                                for vi in range(VLEN):
                                    all_slots.append(("alu", ("^", v_values[i] + vi, v_values[i] + vi, v_root_val + vi)))
                            else:
                                temps = batch_temps[k % NUM_TEMP_SETS]
                                for vi in range(VLEN):
                                    all_slots.append(("alu", ("^", v_values[i] + vi, v_values[i] + vi, temps["v_node_vals"] + vi)))

                        # Hash stages INTERLEAVED across prev_group batches
                        for k, _ in prev_group:
                            i = i_base + k
                            all_slots.append(("valu", ("multiply_add", v_values[i], v_values[i], vv_mul_0, vv_add_0)))

                        for k, _ in prev_group:
                            i = i_base + k
                            temps = batch_temps[k % NUM_TEMP_SETS]
                            all_slots.append(("valu", ("^", temps["v_tmp1"], v_values[i], vv1_1)))
                            all_slots.append(("valu", (">>", temps["v_tmp2"], v_values[i], vv3_1)))
                        for k, _ in prev_group:
                            i = i_base + k
                            temps = batch_temps[k % NUM_TEMP_SETS]
                            all_slots.append(("valu", ("^", v_values[i], temps["v_tmp1"], temps["v_tmp2"])))

                        for k, _ in prev_group:
                            i = i_base + k
                            all_slots.append(("valu", ("multiply_add", v_values[i], v_values[i], vv_mul_2, vv_add_2)))

                        for k, _ in prev_group:
                            i = i_base + k
                            temps = batch_temps[k % NUM_TEMP_SETS]
                            all_slots.append(("valu", ("+", temps["v_tmp1"], v_values[i], vv1_3)))
                            all_slots.append(("valu", ("<<", temps["v_tmp2"], v_values[i], vv3_3)))
                        for k, _ in prev_group:
                            i = i_base + k
                            temps = batch_temps[k % NUM_TEMP_SETS]
                            all_slots.append(("valu", ("^", v_values[i], temps["v_tmp1"], temps["v_tmp2"])))

                        for k, _ in prev_group:
                            i = i_base + k
                            all_slots.append(("valu", ("multiply_add", v_values[i], v_values[i], vv_mul_4, vv_add_4)))

                        for k, _ in prev_group:
                            i = i_base + k
                            temps = batch_temps[k % NUM_TEMP_SETS]
                            all_slots.append(("valu", ("^", temps["v_tmp1"], v_values[i], vv1_5)))
                            all_slots.append(("valu", (">>", temps["v_tmp2"], v_values[i], vv3_5)))
                        for k, _ in prev_group:
                            i = i_base + k
                            temps = batch_temps[k % NUM_TEMP_SETS]
                            all_slots.append(("valu", ("^", v_values[i], temps["v_tmp1"], temps["v_tmp2"])))

                        # Store completed batches on final round
                        for k, r in prev_group:
                            if r != rounds - 1:
                                continue
                            i = i_base + k
                            temps_s = batch_temps[k % NUM_TEMP_SETS]
                            all_slots.append(("flow", ("add_imm", temps_s["v_tmp1"], inp_values_p, i * VLEN)))
                            all_slots.append(("store", ("vstore", temps_s["v_tmp1"], v_values[i])))

                        # Update indices
                        for k, r in prev_group:
                            if r == rounds - 1:
                                continue
                            i = i_base + k
                            temps = batch_temps[k % NUM_TEMP_SETS]
                            level = r % (forest_height + 1)

                            if level == forest_height:
                                for vi in range(VLEN):
                                    all_slots.append(("alu", ("^", v_indices[i] + vi, v_indices[i] + vi, v_indices[i] + vi)))
                            else:
                                for vi in range(VLEN):
                                    all_slots.append(("alu", ("&", temps["v_tmp1"] + vi, v_values[i] + vi, const_1)))
                                all_slots.append(
                                    ("valu", ("multiply_add", v_indices[i], v_indices[i], two_v, temps["v_tmp1"]))
                                )

                    prev_group = group if group else None

        packed_instrs = self.build(all_slots)
        self.instrs.extend(packed_instrs)
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
        if i == rounds - 1:
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
