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
    LOOK_AHEAD_NUMBER = 1000
    NUM_PARALLEL_BLOCKS = 17

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
        enable_logging = self.enable_bundle_logging

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
            # Use a more aggressive multi-pass approach to find independent operations
            skipped_outputs = set()
            skipped_inputs = set()  # Track what skipped slots read (for WAR hazards)
            made_progress = True
            while made_progress:
                made_progress = False
                for look_ahead in range(idx, min(n, idx+self.LOOK_AHEAD_NUMBER)):
                    if used[look_ahead]:
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
                        skipped_inputs.update(look_ahead_inputs)
                        continue
                    # WAW: Can't add if slot writes to a skipped slot's output
                    if look_ahead_outputs & skipped_outputs:
                        log_decision(f"  Lookahead {look_ahead}: WAW hazard (outputs {look_ahead_outputs} & skipped_outputs {skipped_outputs}), skipping.")
                        skipped_outputs.update(look_ahead_outputs)
                        skipped_inputs.update(look_ahead_inputs)
                        continue
                    # WAR: Can't add if slot writes to something a skipped slot reads
                    if look_ahead_outputs & skipped_inputs:
                        log_decision(f"  Lookahead {look_ahead}: WAR hazard (outputs {look_ahead_outputs} & skipped_inputs {skipped_inputs}), skipping.")
                        skipped_outputs.update(look_ahead_outputs)
                        skipped_inputs.update(look_ahead_inputs)
                        continue
                    if can_add_to_bundle(look_ahead_engine, look_ahead_inputs, look_ahead_outputs):
                        log_decision(f"  Lookahead {look_ahead}: can add to bundle: engine={look_ahead_engine}, slot={look_ahead_slot}.")
                        bundle.setdefault(look_ahead_engine, []).append(look_ahead_slot)
                        bundle_input.update(look_ahead_inputs)
                        bundle_output.update(look_ahead_outputs)
                        used[look_ahead] = True
                        made_progress = True
                    else:
                        log_decision(f"  Lookahead {look_ahead}: cannot add to bundle, updating skipped_outputs (outputs {look_ahead_outputs}).")
                        skipped_outputs.update(look_ahead_outputs)
                        skipped_inputs.update(look_ahead_inputs)

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
        # Scratch space addresses
        n_chunks = batch_size // VLEN
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
        for i, v in enumerate(init_vars):
            self.scratch[v] = header_base + i
            self.scratch_debug[header_base + i] = (v, 1) 
        idx_cache = self.alloc_scratch("idx_cache", batch_size)
        val_cache = self.alloc_scratch("val_cache", batch_size)
        v_n_nodes = self.alloc_scratch("v_n_nodes", VLEN)
        v_forest_p = self.alloc_scratch("v_forest_p", VLEN)
        idx_addrs = [self.alloc_scratch(f"idx_addr_{i}") for i in range(n_chunks)]
        val_addrs = [self.alloc_scratch(f"val_addr_{i}") for i in range(n_chunks)]
        # Note: val_store_addrs would be same as val_addrs, so we reuse val_addrs for stores

        scratch_blocks = []
        for i in range(self.NUM_PARALLEL_BLOCKS):
            scratch_blocks.append({
                "node_val": self.alloc_scratch(f"block_{i}_node_val", VLEN),
                "addr": self.alloc_scratch(f"block_{i}_addr", VLEN),
                "vtmp1": self.alloc_scratch(f"block_{i}_vtmp1", VLEN),
                "vtmp2": self.alloc_scratch(f"block_{i}_vtmp2", VLEN),
                "vtmp3": self.alloc_scratch(f"block_{i}_vtmp3", VLEN),
            })

        forest_values_level_0_2 = self.alloc_scratch("forest_values_level_0_2", 8)
        
        # Pre-allocate vector registers for preloaded tree values (levels 0-2)
        # Node 0 (level 0), nodes 1-2 (level 1), nodes 3-6 (level 2) = 7 nodes
        v_node_vals = [self.alloc_scratch(f"v_node_{i}", VLEN) for i in range(7)]
        
        # Collect init slots and build them together for bundling
        init_slots = []

        zero_const = self.scratch_const(0, init_slots=init_slots)
        one_const = self.scratch_const(1, init_slots=init_slots)
        # Pre-allocate offset constants and address registers for parallel loading
        offset_consts = [self.scratch_const(i * VLEN, init_slots=init_slots) for i in range(n_chunks)]
        
        vone_const = self.scratch_vconst(1, "vone", init_slots=init_slots)
        vtwo_const = self.scratch_vconst(2, "vtwo", init_slots=init_slots)
        
        # Load header first
        init_slots.append(("load", ("vload", header_base, zero_const)))
        
        # Compute all addresses using ALU (12 slots/cycle) - removes flow bottleneck
        # These can all be bundled together after header loads
        for j in range(n_chunks):
            init_slots.append(("alu", ("+", idx_addrs[j], self.scratch["inp_indices_p"], offset_consts[j])))
            init_slots.append(("alu", ("+", val_addrs[j], self.scratch["inp_values_p"], offset_consts[j])))
        
        # Interleave cache loads with broadcasts and other VALU work for better bundling
        # First, emit the broadcasts that depend on header (loaded above)
        init_slots.append(("valu", ("vbroadcast", v_n_nodes, self.scratch["n_nodes"])))
        init_slots.append(("valu", ("vbroadcast", v_forest_p, self.scratch["forest_values_p"])))

        # Pre-allocate hash stage constants (these add VALU ops that can overlap with loads)
        for stage_idx in range(len(HASH_STAGES)):
            op1, val1, op2, op3, val3 = HASH_STAGES[stage_idx]
            if op1 == "+" and op2 == "+" and op3 == "<<":
                factor = 1 + (1 << val3)
                self.scratch_vconst(factor, init_slots=init_slots)
                self.scratch_vconst(val1, init_slots=init_slots)
            else:
                self.scratch_vconst(val1, init_slots=init_slots)
                self.scratch_vconst(val3, init_slots=init_slots)

        # Load tree values for levels 0-2 BEFORE main cache loads (so broadcasts can overlap)
        init_slots.append(("load", ("vload", forest_values_level_0_2, self.scratch["forest_values_p"])))

        # Preload tree node values as vectors - these broadcasts can overlap with cache loads
        for node_idx in range(7):
            init_slots.append(("valu", ("vbroadcast", v_node_vals[node_idx], forest_values_level_0_2 + node_idx)))

        # NOW issue the main cache loads (64 vloads) - bundler can overlap with remaining VALU work
        for j in range(n_chunks):
            init_slots.append(("load", ("vload", idx_cache + j * VLEN, idx_addrs[j])))
            init_slots.append(("load", ("vload", val_cache + j * VLEN, val_addrs[j])))

        # Unify init_slots and body for combined bundling
        body = init_slots  # Continue adding to the same list

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
        
        def emit_stage0_addr_compute(iter_idx):
            """Stage 0: Pre-compute addresses for scattered loads (levels 3+)."""
            params = get_iter_params(iter_idx)
            if params is None:
                return []
            round_num, i = params
            block = get_block(iter_idx)
            vtmp_idx = idx_cache + i
            vtmp_addr = block["addr"]
            
            level = round_num % (forest_height + 1)
            slots = []
            if level >= 3:
                slots.append(("valu", ("+", vtmp_addr, v_forest_p, vtmp_idx)))
            return slots
        
        def emit_scalar_xor_level012(iter_idx):
            """
            Emit 8 scalar ALU XOR operations for levels 0, 1, 2.
            This uses ALU slots instead of VALU for the XOR, allowing parallel 
            execution with VALU operations from other iterations.
            
            For level 0: node value is fixed at forest_values_level_0_2
            For levels 1, 2: node values are in block["node_val"] (computed by vselect)
            """
            params = get_iter_params(iter_idx)
            if params is None:
                return []
            round_num, batch_offset = params
            block = get_block(iter_idx)
            
            level = round_num % (forest_height + 1)
            if level > 2:
               return []  # Only handle levels 0, 1, 2
            
            slots = []

            # Special case: use a single VALU XOR for the first chunk
            if batch_offset % 2 == 0:
                vtmp_val = val_cache + batch_offset
                if level == 0:
                    slots.append(("valu", ("^", vtmp_val, vtmp_val, v_node_vals[0])))
                else:
                    slots.append(("valu", ("^", vtmp_val, vtmp_val, block["node_val"])))
                return slots
            
            # Use 8 ALU XORs
            for j in range(VLEN):
                val_addr = val_cache + batch_offset + j
                if level == 0:
                    # Level 0: XOR with the preloaded node 0 value (fixed)
                    slots.append(("alu", ("^", val_addr, val_addr, forest_values_level_0_2)))
                else:
                    # Levels 1, 2: XOR with node value from vselect result
                    node_val_addr = block["node_val"] + j
                    slots.append(("alu", ("^", val_addr, val_addr, node_val_addr)))
            
            return slots
        
        def emit_scalar_idx_update_level012(iter_idx):
            """
            Emit scalar ALU index update for levels 0, 1, 2.
            Uses ALU slots instead of VALU for idx calculation.
            
            idx = 2 * idx + (val & 1) + 1
            For level 0 (idx starts at 0): new_idx = (val & 1) + 1
            For levels 1, 2: new_idx = 2 * idx + (val & 1) + 1
            """
            params = get_iter_params(iter_idx)
            if params is None:
                return []
            round_num, batch_offset = params
            block = get_block(iter_idx)
            
            level = round_num % (forest_height + 1)
            if level > 2:
                return []  # Only handle levels 0, 1, 2
            
            # Skip idx update on last round
            if round_num == rounds - 1:
                return []
            
            slots = []

            # Special case: use VALU for the first chunk
            if batch_offset == 0:
                vtmp_val = val_cache + batch_offset
                vtmp_idx = idx_cache + batch_offset
                if level == 0:
                    slots.append(("valu", ("&", block["vtmp1"], vtmp_val, vone_const)))
                    slots.append(("valu", ("+", vtmp_idx, block["vtmp1"], vone_const)))
                else:
                    slots.append(("valu", ("&", block["vtmp1"], vtmp_val, vone_const)))
                    slots.append(("valu", ("+", block["vtmp1"], block["vtmp1"], vone_const)))
                    slots.append(("valu", ("+", block["vtmp2"], vtmp_idx, vtmp_idx)))
                    slots.append(("valu", ("+", vtmp_idx, block["vtmp2"], block["vtmp1"])))
                return slots
            
            for j in range(VLEN):
                val_addr = val_cache + batch_offset + j
                idx_addr = idx_cache + batch_offset + j
                tmp1_addr = block["vtmp1"] + j
                tmp2_addr = block["vtmp2"] + j
                
                if level == 0:
                    # idx starts at 0: new_idx = (val & 1) + 1
                    slots.append(("alu", ("&", tmp1_addr, val_addr, one_const)))
                    slots.append(("alu", ("+", idx_addr, tmp1_addr, one_const)))
                else:
                    # General case: new_idx = 2 * idx + (val & 1) + 1
                    # tmp1 = val & 1
                    slots.append(("alu", ("&", tmp1_addr, val_addr, one_const)))
                    # tmp1 = tmp1 + 1
                    slots.append(("alu", ("+", tmp1_addr, tmp1_addr, one_const)))
                    # tmp2 = idx + idx (= 2 * idx)
                    slots.append(("alu", ("+", tmp2_addr, idx_addr, idx_addr)))
                    # idx = tmp2 + tmp1
                    slots.append(("alu", ("+", idx_addr, tmp2_addr, tmp1_addr)))
            
            return slots
        
        def emit_stage1_loads_node_val(iter_idx):
            """Stage 1: loads tree node values.
            
            For levels 0, 1, 2: use preloaded values with vselect
            For levels 3+: use scattered loads
            """
            params = get_iter_params(iter_idx)
            if params is None:
                return []
            round_num, i = params
            block = get_block(iter_idx)
            vtmp_idx = idx_cache + i
            vtmp_addr = block["addr"]
            vtmp_node_val = block["node_val"]
            vtmp1 = block["vtmp1"]
            vtmp2 = block["vtmp2"]
            vtmp3 = block["vtmp3"]
            
            level = round_num % (forest_height + 1)
            slots = []
            
            if level == 0:
                # Level 0: handled by scalar ALU XOR, no node_val setup needed
                pass
            elif level == 1:
                # Level 1: 2 nodes (idx=1,2) - use preloaded vectors and vselect
                slots.append(("valu", ("==", vtmp1, vtmp_idx, vone_const)))
                slots.append(("flow", ("vselect", vtmp_node_val, vtmp1, v_node_vals[1], v_node_vals[2])))
            elif level == 2:
                # Level 2: 4 nodes (idx=3,4,5,6) - use preloaded vectors and nested vselect
                slots.append(("valu", ("&", vtmp1, vtmp_idx, vone_const)))
                vfive_const = self.scratch_vconst(5)
                slots.append(("valu", ("<", vtmp2, vtmp_idx, vfive_const)))
                slots.append(("flow", ("vselect", vtmp3, vtmp1, v_node_vals[3], v_node_vals[4])))
                slots.append(("flow", ("vselect", vtmp_node_val, vtmp1, v_node_vals[5], v_node_vals[6])))
                slots.append(("flow", ("vselect", vtmp_node_val, vtmp2, vtmp3, vtmp_node_val)))
            else:
                # For levels 3+, use scattered load (address computed in stage 0)
                for j in range(VLEN):
                    slots.append(("load", ("load_offset", vtmp_node_val, vtmp_addr, j)))
            
            if self.enable_debug and level > 0:
                slots.append(("debug", ("vcompare", vtmp_node_val, [(round_num, i+j, "node_val") for j in range(VLEN)])))
            return slots
        
        def emit_stage2_xor(iter_idx):
            """Stage 2: XOR.
            
            Levels 0, 1, 2: handled by emit_scalar_xor_level012 using ALU
            Levels 3+: use VALU
            """
            params = get_iter_params(iter_idx)
            if params is None:
                return []
            round_num, i = params
            
            level = round_num % (forest_height + 1)
            
            # Levels 0, 1, 2 are handled by scalar ALU XOR
            if level <= 2:
                return []
            
            block = get_block(iter_idx)
            vtmp_val = val_cache + i
            vtmp_node_val = block["node_val"]
            
            slots = []
            slots.append(("valu", ("^", vtmp_val, vtmp_val, vtmp_node_val)))
            return slots
        
        def emit_hash_part1(iter_idx):
            """Hash stages 0-2 (first half)."""
            params = get_iter_params(iter_idx)
            if params is None:
                return []
            round_num, i = params
            block = get_block(iter_idx)
            vtmp_val = val_cache + i
            slots = []
            # First 3 hash stages
            for hi in range(3):
                op1, val1, op2, op3, val3 = HASH_STAGES[hi]
                if op1 == "+" and op2 == "+" and op3 == "<<":
                    factor = 1 + (1 << val3)
                    slots.append(("valu", ("multiply_add", vtmp_val, vtmp_val,
                                          self.scratch_vconst(factor), self.scratch_vconst(val1))))
                else:
                    slots.append(("valu", (op1, block["vtmp1"], vtmp_val, self.scratch_vconst(val1))))
                    slots.append(("valu", (op3, block["vtmp2"], vtmp_val, self.scratch_vconst(val3))))
                    slots.append(("valu", (op2, vtmp_val, block["vtmp1"], block["vtmp2"])))
            return slots

        def emit_hash_part2(iter_idx):
            """Hash stages 3-5 + idx_update (second half).
            
            Idx update for levels 0, 1, 2 is handled by emit_scalar_idx_update_level012 using ALU.
            """
            params = get_iter_params(iter_idx)
            if params is None:
                return []
            round_num, i = params
            block = get_block(iter_idx)
            vtmp_idx = idx_cache + i
            vtmp_val = val_cache + i
            slots = []
            # Last 3 hash stages
            for hi in range(3, 6):
                op1, val1, op2, op3, val3 = HASH_STAGES[hi]
                if op1 == "+" and op2 == "+" and op3 == "<<":
                    factor = 1 + (1 << val3)
                    slots.append(("valu", ("multiply_add", vtmp_val, vtmp_val,
                                          self.scratch_vconst(factor), self.scratch_vconst(val1))))
                else:
                    slots.append(("valu", (op1, block["vtmp1"], vtmp_val, self.scratch_vconst(val1))))
                    slots.append(("valu", (op3, block["vtmp2"], vtmp_val, self.scratch_vconst(val3))))
                    slots.append(("valu", (op2, vtmp_val, block["vtmp1"], block["vtmp2"])))
            if self.enable_debug:
                slots.append(("debug", ("vcompare", vtmp_val, [(round_num, i+j, "hashed_val") for j in range(VLEN)])))

            # Determine current level to optimize idx calculation
            # Levels 0, 1, 2 idx update is handled by scalar ALU path
            level = round_num % (forest_height + 1)
            if round_num != rounds - 1 and level > 2:
                if level < forest_height:
                    slots.append(("valu", ("&", block["vtmp3"], vtmp_val, vone_const)))
                    slots.append(("valu", ("+", block["vtmp3"], block["vtmp3"], vone_const)))
                    slots.append(("valu", ("multiply_add", vtmp_idx, vtmp_idx, vtwo_const, block["vtmp3"])))
                else:
                    slots.append(("valu", ("^", vtmp_idx, vtmp_idx, vtmp_idx)))

            if self.enable_debug:
                slots.append(("debug", ("vcompare", vtmp_idx, [(round_num, i+j, "wrapped_idx") for j in range(VLEN)])))
            return slots
        
        # Software pipelined execution with 5 stages (hash split into 2 parts)
        # Process 2 iterations per step to give bundler more operations to work with
        n_iters = rounds * (batch_size // VLEN)
        NUM_STAGES = 5
        # Compute safe UNROLL based on available parallel blocks
        # Max UNROLL is NUM_PARALLEL_BLOCKS // 2 to avoid aliasing
        MAX_UNROLL = self.NUM_PARALLEL_BLOCKS // 2
        UNROLL = min(MAX_UNROLL, 8)  # Cap at 8 for reasonable code size
        assert UNROLL >= 1, f"NUM_PARALLEL_BLOCKS ({self.NUM_PARALLEL_BLOCKS}) too small, need at least 2"
        total_steps = (n_iters + UNROLL - 1) // UNROLL + NUM_STAGES - 1

        for step in range(total_steps):
            # Process UNROLL iterations at each pipeline stage

            # Stage 0: Address compute + Load node val
            for u in range(UNROLL):
                iter_s0 = step * UNROLL + u
                if 0 <= iter_s0 < n_iters:
                    body.extend(emit_stage0_addr_compute(iter_s0))
                    body.extend(emit_stage1_loads_node_val(iter_s0))

            # Stage 1: XOR
            # For levels 0, 1, 2, 3: use scalar ALU XOR (8 ops in parallel)
            # For levels 4+: use VALU XOR
            for u in range(UNROLL):
                iter_s1 = (step - 1) * UNROLL + u
                if 0 <= iter_s1 < n_iters:
                    body.extend(emit_scalar_xor_level012(iter_s1))
                    body.extend(emit_stage2_xor(iter_s1))

            # Stage 2: Hash part 1 (stages 0-2)
            for u in range(UNROLL):
                iter_s2 = (step - 2) * UNROLL + u
                if 0 <= iter_s2 < n_iters:
                    body.extend(emit_hash_part1(iter_s2))

            # Stage 3: Hash part 2 (stages 3-5) + Idx update
            # For levels 0, 1, 2: idx update uses scalar ALU
            # For levels 4+: idx update uses VALU
            for u in range(UNROLL):
                iter_s3 = (step - 3) * UNROLL + u
                if 0 <= iter_s3 < n_iters:
                    body.extend(emit_hash_part2(iter_s3))
                    body.extend(emit_scalar_idx_update_level012(iter_s3))
            
        # Issue all stores - only val needed for correctness check
        # Reuse val_addrs (computed in init) since they equal inp_values_p + offset
        for j in range(n_chunks):
            body.append(("store", ("vstore", val_addrs[j], val_cache + j * VLEN)))

        body_instrs = self.build(body)
        self.instrs.extend(body_instrs)
        with open("kernel_instr_counts.txt", "w") as f:
            for instr in self.instrs:        
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
