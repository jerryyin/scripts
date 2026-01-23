#!/usr/bin/env python3
"""
LDS Bank Conflict Analyzer for AMD GPUs

This script analyzes bank conflicts for shared memory (LDS) access patterns,
particularly useful for understanding ds_load_tr* instructions and WMMA operand loads.

=============================================================================
CONCEPTS
=============================================================================

1. LDS Banks:
   - AMD gfx1250 has 64 banks, each 4 bytes wide
   - Bank = (byte_address / 4) % 64
   - Bank conflict occurs when multiple lanes access the same bank simultaneously

2. Linear Layout (from Triton TTGIR):
   - Describes how (register_idx, lane_id, warp_id) maps to tensor [row, col]
   - Format: lane = [[row_contrib, col_contrib], ...] for each bit of lane ID
   - Example: lane = [[1,0], [2,0], [4,0], [8,0], [0,8]]
     - Bits 0-3 contribute to row (1, 2, 4, 8)
     - Bit 4 contributes to column (8)

3. ds_load_tr16_b128:
   - Each lane loads 16 bytes (8 fp16 elements)
   - 8 lanes cooperate for transposed load
   - Accesses 4 consecutive banks per lane

=============================================================================
USAGE
=============================================================================

    python3 lds_bank_conflict_analyzer.py

Or import as module:

    from lds_bank_conflict_analyzer import analyze_bank_conflicts, LDSConfig, AccessPattern

=============================================================================
"""

from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import argparse


@dataclass
class LDSConfig:
    """
    Configuration for LDS memory layout.
    
    Attributes:
        num_banks: Number of LDS banks (64 for gfx1250)
        bytes_per_bank: Bytes per bank (4 for AMD GPUs)
        row_width_elements: Number of data elements per row (before padding)
        padding_elements: Number of padding elements after each row
        element_bytes: Bytes per element (2 for fp16, 4 for fp32)
    """
    num_banks: int = 64
    bytes_per_bank: int = 4
    row_width_elements: int = 128
    padding_elements: int = 0
    element_bytes: int = 2
    
    @property
    def row_stride_bytes(self) -> int:
        """Total bytes per row including padding."""
        return (self.row_width_elements + self.padding_elements) * self.element_bytes


@dataclass 
class AccessPattern:
    """
    Describes how lanes access tensor positions.
    
    Attributes:
        lanes_per_wave: Number of lanes per wave (32 for AMD)
        elements_per_load: Elements loaded per lane per instruction (8 for ds_load_tr16_b128)
        lane_to_position: List of (row_offset, col_offset) for each lane
                         Derived from linear layout's lane field
        element_offsets: Column offsets for each of the elements_per_load elements
                        Derived from register bits. Default: [0, 1, 2, ..., 7] (consecutive)
                        
    Note on linear layout interpretation:
        - lane bits: Define the STARTING [row, col] for each lane
        - register bits 0-N: Define the ELEMENT positions within one load
        - Higher register bits: Define tiling across the full tensor (not used here)
        
    For ds_load_tr16_b128 with register = [[0,1], [0,2], [0,4], ...]:
        - Bits 0-2 contribute cols 1, 2, 4 -> elements at consecutive cols 0-7
        - This is the "basic unit" that one instruction loads
    """
    lanes_per_wave: int = 32
    elements_per_load: int = 8
    lane_to_position: List[Tuple[int, int]] = None
    element_offsets: List[int] = None  # Column offset for each element within the load
    
    def __post_init__(self):
        if self.lane_to_position is None:
            # Default: simple row-major access
            self.lane_to_position = [(i, 0) for i in range(self.lanes_per_wave)]
        if self.element_offsets is None:
            # Default: consecutive elements [0, 1, 2, ..., elements_per_load-1]
            self.element_offsets = list(range(self.elements_per_load))
    
    @classmethod
    def from_linear_layout(cls, lane_bits: List[Tuple[int, int]], 
                           register_bits: List[Tuple[int, int]] = None) -> 'AccessPattern':
        """
        Create AccessPattern from Triton linear layout.
        
        Args:
            lane_bits: List of [row_contribution, col_contribution] for each lane bit
                      e.g., [[1,0], [2,0], [4,0], [8,0], [0,8]] from TTGIR
                      Number of lanes = 2^len(lane_bits)
            register_bits: List of [row_contribution, col_contribution] for register bits
                          that define element positions within one load.
                          e.g., [[0,1], [0,2], [0,4]] for 8 consecutive elements
                          Number of elements = 2^len(register_bits)
                          If None, assumes 8 consecutive elements (default for ds_load_tr16_b128).
        
        Returns:
            AccessPattern with lane_to_position and element_offsets computed
        
        Note:
            lanes_per_wave and elements_per_load are DERIVED from the bit counts:
            - lanes_per_wave = 2^len(lane_bits)
            - elements_per_load = 2^len(register_bits)
        """
        # Derive counts from bit lengths
        num_lane_bits = len(lane_bits)
        lanes_per_wave = 1 << num_lane_bits  # 2^num_lane_bits
        
        if register_bits is not None:
            num_reg_bits = len(register_bits)
            elements_per_load = 1 << num_reg_bits  # 2^num_reg_bits
        else:
            # Default: 8 elements (3 bits) with consecutive columns
            num_reg_bits = 3
            elements_per_load = 8
            register_bits = [[0, 1], [0, 2], [0, 4]]  # Consecutive cols 0-7
        
        # Compute lane starting positions
        lane_to_position = []
        for lane_id in range(lanes_per_wave):
            row = 0
            col = 0
            for bit_idx in range(num_lane_bits):
                if lane_id & (1 << bit_idx):
                    row += lane_bits[bit_idx][0]
                    col += lane_bits[bit_idx][1]
            lane_to_position.append((row, col))
        
        # Compute element offsets within each load from register bits
        element_offsets = []
        for elem_idx in range(elements_per_load):
            col_offset = 0
            for bit_idx in range(num_reg_bits):
                if elem_idx & (1 << bit_idx):
                    col_offset += register_bits[bit_idx][1]
            element_offsets.append(col_offset)
        
        return cls(lanes_per_wave=lanes_per_wave,
                   elements_per_load=elements_per_load,
                   lane_to_position=lane_to_position,
                   element_offsets=element_offsets)


def compute_banks_accessed(byte_addr: int, 
                           load_size_bytes: int,
                           num_banks: int = 64,
                           bytes_per_bank: int = 4) -> List[int]:
    """
    Compute which banks are accessed for a load starting at byte_addr.
    
    Args:
        byte_addr: Starting byte address in LDS
        load_size_bytes: Number of bytes loaded (16 for ds_load_tr16_b128)
        num_banks: Number of LDS banks
        bytes_per_bank: Bytes per bank
    
    Returns:
        List of bank indices accessed
    """
    num_banks_accessed = load_size_bytes // bytes_per_bank
    start_bank = (byte_addr // bytes_per_bank) % num_banks
    return [(start_bank + i) % num_banks for i in range(num_banks_accessed)]


def compute_lane_accesses(config: LDSConfig, 
                          pattern: AccessPattern) -> List[Dict]:
    """
    Compute LDS access details for each lane.
    
    Args:
        config: LDS configuration
        pattern: Access pattern
    
    Returns:
        List of dicts with lane access info:
        [{lane, row, col, byte_addr, banks, element_addrs}, ...]
    """
    accesses = []
    
    for lane_id in range(pattern.lanes_per_wave):
        row, col_start = pattern.lane_to_position[lane_id]
        
        # Compute address and banks for each element in the load
        element_addrs = []
        all_banks = set()
        
        for elem_offset in pattern.element_offsets:
            col = col_start + elem_offset
            byte_addr = row * config.row_stride_bytes + col * config.element_bytes
            element_addrs.append(byte_addr)
            bank = (byte_addr // config.bytes_per_bank) % config.num_banks
            all_banks.add(bank)
        
        # First element address (for display)
        base_addr = element_addrs[0]
        # Banks accessed (sorted for consistent output)
        banks = sorted(all_banks)
        
        accesses.append({
            'lane': lane_id,
            'row': row,
            'col': col_start,
            'byte_addr': base_addr,
            'banks': banks,
            'element_addrs': element_addrs
        })
    
    return accesses


def find_bank_conflicts(accesses: List[Dict]) -> Tuple[Dict[int, List[int]], int]:
    """
    Find bank conflicts from lane accesses.
    
    Args:
        accesses: List of lane access info from compute_lane_accesses()
    
    Returns:
        Tuple of (bank_to_lanes dict, max_conflict_count)
    """
    bank_to_lanes = {}
    
    for access in accesses:
        for bank in access['banks']:
            if bank not in bank_to_lanes:
                bank_to_lanes[bank] = []
            bank_to_lanes[bank].append(access['lane'])
    
    max_conflict = max(len(lanes) for lanes in bank_to_lanes.values()) if bank_to_lanes else 0
    
    return bank_to_lanes, max_conflict


def generate_bank_grid(config: LDSConfig, 
                       tile_rows: int, 
                       tile_cols: int) -> List[List[int]]:
    """
    Generate a grid showing which bank each element belongs to.
    
    Args:
        config: LDS configuration
        tile_rows: Number of rows in the tile
        tile_cols: Number of columns in the tile
    
    Returns:
        2D list of bank indices [row][col]
    """
    grid = []
    for row in range(tile_rows):
        row_banks = []
        for col in range(tile_cols):
            byte_addr = row * config.row_stride_bytes + col * config.element_bytes
            bank = (byte_addr // config.bytes_per_bank) % config.num_banks
            row_banks.append(bank)
        grid.append(row_banks)
    return grid


def print_bank_grid(grid: List[List[int]], max_cols: int = 16):
    """Print bank grid in readable format."""
    num_rows = len(grid)
    num_cols = len(grid[0]) if grid else 0
    
    # Header
    print("     ", end="")
    for col in range(min(num_cols, max_cols)):
        print(f"{col:3d}", end="")
    if num_cols > max_cols:
        print(" ...", end="")
    print("   ← column")
    print("     " + "─" * (min(num_cols, max_cols) * 3 + 4))
    
    # Grid
    for row in range(num_rows):
        print(f"r{row:2d} │", end="")
        for col in range(min(num_cols, max_cols)):
            print(f"{grid[row][col]:3d}", end="")
        if num_cols > max_cols:
            print(" ...", end="")
        print()


def print_lane_accesses(accesses: List[Dict]):
    """Print lane access table."""
    print("Lane | Row | Col | Addr (bytes) | Banks accessed")
    print("-----|-----|-----|--------------|----------------")
    for a in accesses:
        print(f" {a['lane']:2d}  |  {a['row']:2d} | {a['col']:2d}  |    {a['byte_addr']:5d}     | {a['banks']}")


def print_conflict_summary(bank_to_lanes: Dict[int, List[int]], max_conflict: int):
    """Print bank conflict summary."""
    if max_conflict <= 1:
        print("✓ No bank conflicts!")
    else:
        print(f"✗ Max conflict: {max_conflict}-way")
        print()
        print("Conflicting banks (first 4):")
        conflict_banks = [(b, lanes) for b, lanes in bank_to_lanes.items() 
                         if len(lanes) == max_conflict]
        for bank, lanes in conflict_banks[:4]:
            print(f"  Bank {bank:2d}: lanes {lanes}")


def analyze_bank_conflicts(config: LDSConfig, 
                           pattern: AccessPattern,
                           tile_rows: int = 16,
                           tile_cols: int = 16,
                           verbose: bool = True) -> int:
    """
    Full bank conflict analysis.
    
    Args:
        config: LDS configuration
        pattern: Access pattern
        tile_rows: Rows in tile to visualize
        tile_cols: Columns in tile to visualize  
        verbose: Print detailed output
    
    Returns:
        Maximum conflict count
    """
    if verbose:
        print("=" * 70)
        print("BANK CONFLICT ANALYSIS")
        print("=" * 70)
        print()
        print(f"LDS Configuration:")
        print(f"  Row width: {config.row_width_elements} elements")
        print(f"  Padding: {config.padding_elements} elements ({config.padding_elements * config.element_bytes} bytes)")
        print(f"  Row stride: {config.row_stride_bytes} bytes")
        print(f"  Element size: {config.element_bytes} bytes")
        print(f"  Banks: {config.num_banks}")
        print()
        print(f"Access Pattern:")
        print(f"  Lanes per wave: {pattern.lanes_per_wave}")
        print(f"  Elements per load: {pattern.elements_per_load}")
        print(f"  Load size: {pattern.elements_per_load * config.element_bytes} bytes")
        print()
    
    # Generate bank grid
    if verbose:
        print("=" * 70)
        print(f"BANK MAPPING ({tile_rows}×{tile_cols} tile)")
        print("=" * 70)
        print()
        grid = generate_bank_grid(config, tile_rows, tile_cols)
        print_bank_grid(grid, max_cols=tile_cols)
        print()
    
    # Compute lane accesses
    accesses = compute_lane_accesses(config, pattern)
    
    if verbose:
        print("=" * 70)
        print("LANE ACCESS PATTERN")
        print("=" * 70)
        print()
        print_lane_accesses(accesses)
        print()
    
    # Find conflicts
    bank_to_lanes, max_conflict = find_bank_conflicts(accesses)
    
    if verbose:
        print("=" * 70)
        print("CONFLICT SUMMARY")
        print("=" * 70)
        print()
        print_conflict_summary(bank_to_lanes, max_conflict)
        print()
    
    return max_conflict


# =============================================================================
# Preset configurations for common use cases
# =============================================================================

def ds_load_tr16_b128_pattern() -> AccessPattern:
    """
    Access pattern for ds_load_tr16_b128 with typical WMMA layout.
    
    From TTGIR #linear layout:
        lane = [[1,0], [2,0], [4,0], [8,0], [0,8]]
        register = [[0,1], [0,2], [0,4], ...]  (first 3 bits for 8 elements)
    
    Lane mapping (derived: 5 bits → 32 lanes):
        - Lanes 0-15: rows 0-15, col 0
        - Lanes 16-31: rows 0-15, col 8
    
    Register mapping (derived: 3 bits → 8 elements):
        - Each lane loads 8 consecutive column elements
        - Element offsets: 0, 1, 2, 3, 4, 5, 6, 7
    """
    return AccessPattern.from_linear_layout(
        lane_bits=[[1, 0], [2, 0], [4, 0], [8, 0], [0, 8]],
        register_bits=[[0, 1], [0, 2], [0, 4]]
    )


def create_config_with_padding(padding_elements: int, 
                                row_width: int = 128,
                                element_bytes: int = 2) -> LDSConfig:
    """Create LDS config with specified padding."""
    return LDSConfig(
        row_width_elements=row_width,
        padding_elements=padding_elements,
        element_bytes=element_bytes
    )


# =============================================================================
# Main entry point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Analyze LDS bank conflicts for AMD GPU shared memory access",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze ds_load_tr16_b128 with 8-element padding
  python3 lds_bank_conflict_analyzer.py --padding 8
  
  # Analyze with custom row width
  python3 lds_bank_conflict_analyzer.py --row-width 64 --padding 8
  
  # Compare multiple padding values
  python3 lds_bank_conflict_analyzer.py --compare
        """
    )
    
    parser.add_argument('--row-width', type=int, default=128,
                        help='Elements per row (default: 128)')
    parser.add_argument('--padding', type=int, default=0,
                        help='Padding elements after each row (default: 0)')
    parser.add_argument('--element-bytes', type=int, default=2,
                        help='Bytes per element (default: 2 for fp16)')
    parser.add_argument('--tile-rows', type=int, default=16,
                        help='Tile rows to visualize (default: 16)')
    parser.add_argument('--tile-cols', type=int, default=16,
                        help='Tile columns to visualize (default: 16)')
    parser.add_argument('--compare', action='store_true',
                        help='Compare padding strategies (0, 8, 16)')
    parser.add_argument('--quiet', action='store_true',
                        help='Only print summary')
    
    args = parser.parse_args()
    
    # Use ds_load_tr16_b128 pattern as default
    pattern = ds_load_tr16_b128_pattern()
    
    if args.compare:
        print("=" * 70)
        print("COMPARING PADDING STRATEGIES")
        print("=" * 70)
        print()
        
        results = []
        for padding in [0, 8, 16]:
            config = create_config_with_padding(padding, args.row_width, args.element_bytes)
            max_conflict = analyze_bank_conflicts(
                config, pattern, 
                tile_rows=args.tile_rows,
                tile_cols=args.tile_cols,
                verbose=not args.quiet
            )
            results.append((padding, config.row_stride_bytes, max_conflict))
            print()
        
        print("=" * 70)
        print("COMPARISON SUMMARY")
        print("=" * 70)
        print()
        print("Padding | Row Stride | Max Conflict")
        print("--------|------------|-------------")
        for padding, stride, conflict in results:
            status = "OK" if conflict <= 2 else "BAD"
            print(f"   {padding:2d}   |    {stride:3d}     |   {conflict:2d}-way {status}")
    else:
        config = create_config_with_padding(args.padding, args.row_width, args.element_bytes)
        max_conflict = analyze_bank_conflicts(
            config, pattern,
            tile_rows=args.tile_rows,
            tile_cols=args.tile_cols,
            verbose=not args.quiet
        )
        if args.quiet:
            status = "OK" if max_conflict <= 2 else "BAD"
            print(f"row={args.row_width} pad={args.padding} stride={config.row_stride_bytes}B "
                  f"gcd={__import__('math').gcd(config.row_stride_bytes, 256)} → {max_conflict}-way {status}")


if __name__ == "__main__":
    main()
