#!/usr/bin/env python3
"""
Parse AMReX plotfile directories and generate YAML descriptions.

This script can parse single or multiple AMReX plotfile directories and generate
a YAML description of the data structure, including grid hierarchy, variables,
and metadata.
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
import re

# Try to import yaml, fall back to JSON if not available
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False
    print("Warning: PyYAML not found. Output will be in JSON format.", file=sys.stderr)
    print("To install PyYAML, run: python3 -m pip install --user pyyaml", file=sys.stderr)


class AMReXPlotfileParser:
    """Parser for AMReX plotfile directories."""
    
    def __init__(self, plotfile_dir: Path):
        self.plotfile_dir = Path(plotfile_dir)
        self.header_path = self.plotfile_dir / "Header"
        
        if not self.header_path.exists():
            raise ValueError(f"No Header file found in {plotfile_dir}")
        
        self.metadata = {}
        self.levels = []
        self.variable_groups = {}
        
    def parse(self) -> Dict:
        """Parse the plotfile and return a dictionary of metadata."""
        self._parse_header()
        self._scan_variable_groups()
        return self._build_description()
    
    def _parse_header(self):
        """Parse the Header file to extract metadata."""
        with open(self.header_path, 'r') as f:
            lines = [line.strip() for line in f.readlines()]
        
        idx = 0
        
        # Line 0: Format version
        self.metadata['format_version'] = lines[idx]
        idx += 1
        
        # Line 1: Number of cell-centered variables
        n_cell_vars = int(lines[idx])
        idx += 1
        
        # Lines 2 to 2+n_cell_vars-1: Variable names
        cell_var_names = []
        for i in range(n_cell_vars):
            cell_var_names.append(lines[idx])
            idx += 1
        self.metadata['cell_centered_vars'] = cell_var_names
        
        # Line: Number of dimensions
        n_dim = int(lines[idx])
        self.metadata['spatial_dimensions'] = n_dim
        idx += 1
        
        # Line: Timestep
        timestep = int(lines[idx])
        self.metadata['timestep'] = timestep
        idx += 1
        
        # Line: max_level (0-indexed, so max_level+1 = num_levels)
        max_level = int(lines[idx])
        n_levels = max_level + 1
        self.metadata['num_levels'] = n_levels
        idx += 1
        
        # Line: Geometry lower corner
        prob_lo = [float(x) for x in lines[idx].split()]
        self.metadata['prob_lo'] = prob_lo
        idx += 1
        
        # Line: Geometry upper corner
        prob_hi = [float(x) for x in lines[idx].split()]
        self.metadata['prob_hi'] = prob_hi
        idx += 1
        
        # Line: Refinement ratios (space-separated, one per refined level)
        ref_ratios = [int(x) for x in lines[idx].split()]
        self.metadata['refinement_ratios'] = ref_ratios
        idx += 1
        
        # Line: Domain boxes for each level (complex format)
        domain_str = lines[idx]
        idx += 1
        
        # Parse domain boxes
        self._parse_domain_boxes(domain_str, n_levels)
        
        # Skip: refinement timesteps (1 line) + cell spacing (n_levels lines) + ngrow (1) + num_field_add_on (1)
        # After reading domain_str and incrementing idx, we need to skip these 4 + n_levels - 1 = n_levels + 3 more lines
        idx += n_levels + 3
        
        # Parse level information
        for level in range(n_levels):
            # Line: level timestep time  
            level_info = lines[idx].split()
            level_num = int(level_info[0])
            timestep = int(level_info[1])
            time = float(level_info[2]) if len(level_info) > 2 else 0.0
            idx += 1
            
            # Line: step number (this appears to be the step count)
            step = int(lines[idx])
            idx += 1
            
            # Lines: level geometry (one line per dimension with lo and hi)
            geom_lo = []
            geom_hi = []
            for dim in range(n_dim):
                parts = lines[idx].split()
                geom_lo.append(float(parts[0]))
                geom_hi.append(float(parts[1]))
                idx += 1
            
            # Line: level directory
            level_dir = lines[idx]
            idx += 1
            
            # Store level info
            self.levels.append({
                'level': level_num,
                'directory': level_dir,
                'geom_lo': geom_lo,
                'geom_hi': geom_hi,
                'timestep': timestep,
                'step': step,
                'time': time
            })
        
        # Remaining lines contain variable group information
        self._parse_variable_groups(lines[idx:])
    
    def _parse_domain_boxes(self, domain_str: str, n_levels: int):
        """Parse the domain box specification."""
        # Format: ((lo,lo,lo) (hi,hi,hi) (0,0,0)) ((lo,lo,lo) (hi,hi,hi) (0,0,0)) ...
        # Extract boxes for each level
        pattern = r'\(\((\d+),(\d+),(\d+)\)\s+\((\d+),(\d+),(\d+)\)'
        matches = re.findall(pattern, domain_str)
        
        self.metadata['domain_boxes'] = []
        for match in matches[:n_levels]:
            lo = [int(match[0]), int(match[1]), int(match[2])]
            hi = [int(match[3]), int(match[4]), int(match[5])]
            self.metadata['domain_boxes'].append({'lo': lo, 'hi': hi})
    
    def _parse_variable_groups(self, lines: List[str]):
        """Parse variable group information from remaining header lines."""
        idx = 0
        
        # Parse additional variable groups
        while idx < len(lines):
            if not lines[idx]:  # Skip empty lines
                idx += 1
                continue
            
            # Number of variables in this group
            try:
                n_vars = int(lines[idx])
            except ValueError:
                idx += 1
                continue
            
            idx += 1
            if idx >= len(lines):
                break
            
            # Variable names
            var_names = []
            for i in range(n_vars):
                if idx >= len(lines):
                    break
                var_names.append(lines[idx])
                idx += 1
            
            # Directory pattern for this group
            if idx < len(lines):
                dir_pattern = lines[idx]
                idx += 1
                
                # Store in temporary structure
                group_key = dir_pattern.split('/')[1] if '/' in dir_pattern else dir_pattern
                if group_key not in self.variable_groups:
                    self.variable_groups[group_key] = {
                        'variables': var_names,
                        'directory_pattern': dir_pattern
                    }
            
            # Skip level directories
            while idx < len(lines) and lines[idx].startswith('Level_'):
                idx += 1
    
    def _scan_variable_groups(self):
        """Scan the plotfile directory to identify all variable groups."""
        for level_num in range(self.metadata['num_levels']):
            level_dir = self.plotfile_dir / f"Level_{level_num}"
            if not level_dir.exists():
                continue
            
            # Find all *_H files
            for header_file in level_dir.glob("*_H"):
                group_name = header_file.stem
                
                # Parse the header file to get dimensions and metadata
                group_info = self._parse_group_header(header_file)
                
                if group_name not in self.variable_groups:
                    self.variable_groups[group_name] = {
                        'variables': [],
                        'directory_pattern': f'Level_*/{group_name}',
                        'levels': {}
                    }
                
                self.variable_groups[group_name]['levels'][level_num] = group_info
    
    def _parse_group_header(self, header_path: Path) -> Dict:
        """Parse a variable group header file (*_H)."""
        with open(header_path, 'r') as f:
            lines = [line.strip() for line in f.readlines()]
        
        info = {}
        idx = 0
        
        # Line 0: ?
        idx += 1
        
        # Line 1: Number of FABs (data arrays)
        n_fabs = int(lines[idx])
        idx += 1
        
        # Line 2: Number of components
        n_comp = int(lines[idx])
        info['num_components'] = n_comp
        idx += 1
        
        # Line 3: Ghost cells or centering info
        if '(' in lines[idx] and ')' in lines[idx]:
            # This is centering information (nodal flags)
            centering = lines[idx].strip('()')
            info['centering'] = [int(x) for x in centering.split(',')]
            idx += 1
        else:
            info['centering'] = [0, 0, 0]  # Cell-centered by default
        
        # Box information
        if idx < len(lines) and lines[idx].startswith('('):
            # Parse box geometry
            box_lines = []
            while idx < len(lines) and (lines[idx].startswith('(') or lines[idx].startswith(')')):
                box_lines.append(lines[idx])
                idx += 1
            
            # Extract box dimensions
            box_str = ' '.join(box_lines)
            box_match = re.search(r'\(\((\d+),(\d+),(\d+)\)\s+\((\d+),(\d+),(\d+)\)', box_str)
            if box_match:
                lo = [int(box_match.group(1)), int(box_match.group(2)), int(box_match.group(3))]
                hi = [int(box_match.group(4)), int(box_match.group(5)), int(box_match.group(6))]
                info['box'] = {'lo': lo, 'hi': hi}
                
                # Calculate dimensions (including centering adjustment)
                dims = [hi[i] - lo[i] + 1 for i in range(3)]
                info['dimensions'] = dims
        
        # Look for min/max values at the end
        min_max_lines = [line for line in lines if line and ',' in line and 'e' in line.lower()]
        if len(min_max_lines) >= 2:
            try:
                min_vals = [float(x) for x in min_max_lines[0].rstrip(',').split(',') if x]
                max_vals = [float(x) for x in min_max_lines[1].rstrip(',').split(',') if x]
                info['min_values'] = min_vals
                info['max_values'] = max_vals
            except (ValueError, IndexError):
                pass
        
        return info
    
    def _build_description(self) -> Dict:
        """Build the final YAML-compatible description dictionary."""
        desc = {
            'plotfile_format': self.metadata.get('format_version', 'AMReX'),
            'description': 'AMReX plotfile data structure',
            'plotfile_path': str(self.plotfile_dir)
        }
        
        # Simulation metadata
        desc['simulation'] = {
            'timestep': self.metadata.get('timestep', 0),
            'spatial_dimensions': self.metadata.get('spatial_dimensions', 3)
        }
        
        # Add time if available
        if self.levels and 'time' in self.levels[0]:
            desc['simulation']['time'] = self.levels[0]['time']
        
        # Domain configuration
        desc['domain'] = {
            'coordinate_system': 'cartesian',
            'physical_extent': {
                'x': [self.metadata['prob_lo'][0], self.metadata['prob_hi'][0]],
                'y': [self.metadata['prob_lo'][1], self.metadata['prob_hi'][1]],
                'z': [self.metadata['prob_lo'][2], self.metadata['prob_hi'][2]],
                'units': 'meters'
            }
        }
        
        # AMR hierarchy
        desc['amr'] = {
            'num_levels': self.metadata['num_levels'],
            'refinement_ratios': self.metadata.get('refinement_ratios', []),
            'levels': []
        }
        
        for level_info in self.levels:
            level_num = level_info['level']
            domain_box = self.metadata['domain_boxes'][level_num] if level_num < len(self.metadata['domain_boxes']) else None
            
            level_desc = {
                'level': level_num,
                'directory': level_info['directory']
            }
            
            if domain_box:
                level_desc['box'] = [domain_box['lo'], domain_box['hi']]
            
            level_desc['physical_extent'] = {
                'x': [level_info['geom_lo'][0], level_info['geom_hi'][0]],
                'y': [level_info['geom_lo'][1], level_info['geom_hi'][1]],
                'z': [level_info['geom_lo'][2], level_info['geom_hi'][2]]
            }
            
            # Calculate cell spacing
            if domain_box:
                dims = [domain_box['hi'][i] - domain_box['lo'][i] + 1 for i in range(3)]
                dx = (level_info['geom_hi'][0] - level_info['geom_lo'][0]) / dims[0] if dims[0] > 0 else 0
                dy = (level_info['geom_hi'][1] - level_info['geom_lo'][1]) / dims[1] if dims[1] > 0 else 0
                dz = (level_info['geom_hi'][2] - level_info['geom_lo'][2]) / dims[2] if dims[2] > 0 else 0
                level_desc['cell_spacing'] = [dx, dy, dz]
            
            desc['amr']['levels'].append(level_desc)
        
        # Variables
        desc['variables'] = {}
        
        for group_name, group_data in self.variable_groups.items():
            group_desc = {
                'directory_pattern': group_data.get('directory_pattern', f'Level_*/{group_name}')
            }
            
            # Determine grid type from centering
            if 'levels' in group_data and group_data['levels']:
                first_level_data = next(iter(group_data['levels'].values()))
                centering = first_level_data.get('centering', [0, 0, 0])
                
                # Classify grid type
                if centering == [0, 0, 0]:
                    group_desc['grid_type'] = 'cell_centered'
                elif centering == [1, 0, 0]:
                    group_desc['grid_type'] = 'x_face_centered'
                elif centering == [0, 1, 0]:
                    group_desc['grid_type'] = 'y_face_centered'
                elif centering == [0, 0, 1]:
                    group_desc['grid_type'] = 'z_face_centered'
                elif centering == [1, 1, 1]:
                    group_desc['grid_type'] = 'node_centered'
                else:
                    group_desc['grid_type'] = f'custom_{centering[0]}_{centering[1]}_{centering[2]}'
                
                group_desc['num_components'] = first_level_data.get('num_components', 0)
                
                # Add dimensions for each level
                dims_by_level = {}
                for level, level_data in group_data['levels'].items():
                    if 'dimensions' in level_data:
                        dims = level_data['dimensions']
                        # Check if 2D (one dimension is 1)
                        if dims[2] == 1:
                            group_desc['dimensionality'] = '2D'
                            dims_by_level[f'level_{level}'] = dims[:2]
                        else:
                            group_desc['dimensionality'] = '3D'
                            dims_by_level[f'level_{level}'] = dims
                
                group_desc['dimensions'] = dims_by_level
            
            # Add variable names
            if group_name == 'Cell':
                group_desc['variables'] = [{'name': name} for name in self.metadata.get('cell_centered_vars', [])]
            else:
                # Try to infer from other sources or leave empty
                group_desc['variables'] = group_data.get('variables', [])
                if not isinstance(group_desc['variables'], list):
                    group_desc['variables'] = []
                if group_desc['variables'] and isinstance(group_desc['variables'][0], str):
                    group_desc['variables'] = [{'name': v} for v in group_desc['variables']]
            
            desc['variables'][group_name] = group_desc
        
        return desc


class AMReXPlotfileBlender:
    """Blend multiple plotfile descriptions into a unified superset."""
    
    def __init__(self):
        self.descriptions = []
        
    def add_plotfile(self, plotfile_path: Path):
        """Parse and add a plotfile to the blend."""
        parser = AMReXPlotfileParser(plotfile_path)
        desc = parser.parse()
        self.descriptions.append(desc)
    
    def blend(self) -> Dict:
        """Create a blended description from all added plotfiles."""
        if not self.descriptions:
            return {}
        
        blended = {
            'plotfile_format': self.descriptions[0].get('plotfile_format', 'AMReX'),
            'description': 'Blended AMReX plotfile data structure (superset)',
            'num_plotfiles_analyzed': len(self.descriptions),
            'plotfiles': [desc.get('plotfile_path', '') for desc in self.descriptions]
        }
        
        # Merge simulation info
        blended['simulation'] = {
            'spatial_dimensions': self.descriptions[0]['simulation']['spatial_dimensions'],
            'timestep_range': self._get_range('simulation', 'timestep')
        }
        
        if 'time' in self.descriptions[0]['simulation']:
            blended['simulation']['time_range'] = self._get_range('simulation', 'time')
        
        # Domain - use first description as reference
        blended['domain'] = self.descriptions[0]['domain']
        
        # AMR - find maximum number of levels
        max_levels = max(desc['amr']['num_levels'] for desc in self.descriptions)
        blended['amr'] = {
            'num_levels': max_levels,
            'num_levels_range': [
                min(desc['amr']['num_levels'] for desc in self.descriptions),
                max(desc['amr']['num_levels'] for desc in self.descriptions)
            ],
            'refinement_ratios': self.descriptions[0]['amr'].get('refinement_ratios', []),
            'levels': []
        }
        
        # Merge level information
        for level_num in range(max_levels):
            level_descs = [desc['amr']['levels'][level_num] 
                          for desc in self.descriptions 
                          if level_num < len(desc['amr']['levels'])]
            
            if level_descs:
                merged_level = {
                    'level': level_num,
                    'directory': level_descs[0]['directory'],
                    'present_in_plotfiles': len(level_descs),
                    'total_plotfiles': len(self.descriptions)
                }
                
                # Add box and extents from first occurrence
                if 'box' in level_descs[0]:
                    merged_level['box'] = level_descs[0]['box']
                if 'physical_extent' in level_descs[0]:
                    merged_level['physical_extent'] = level_descs[0]['physical_extent']
                if 'cell_spacing' in level_descs[0]:
                    merged_level['cell_spacing'] = level_descs[0]['cell_spacing']
                
                blended['amr']['levels'].append(merged_level)
        
        # Merge variables - union of all variable groups
        all_var_groups = set()
        for desc in self.descriptions:
            all_var_groups.update(desc.get('variables', {}).keys())
        
        blended['variables'] = {}
        for var_group in sorted(all_var_groups):
            # Find all descriptions containing this variable group
            group_descs = [desc['variables'][var_group] 
                          for desc in self.descriptions 
                          if var_group in desc.get('variables', {})]
            
            if group_descs:
                merged_group = group_descs[0].copy()
                merged_group['present_in_plotfiles'] = len(group_descs)
                merged_group['total_plotfiles'] = len(self.descriptions)
                
                # Merge dimensions across levels
                all_dims = {}
                for group_desc in group_descs:
                    if 'dimensions' in group_desc:
                        for level_key, dims in group_desc['dimensions'].items():
                            if level_key not in all_dims:
                                all_dims[level_key] = dims
                
                if all_dims:
                    merged_group['dimensions'] = all_dims
                
                blended['variables'][var_group] = merged_group
        
        return blended
    
    def _get_range(self, section: str, key: str) -> Optional[List]:
        """Get min and max values for a numeric field."""
        values = [desc[section][key] for desc in self.descriptions if key in desc.get(section, {})]
        if values:
            return [min(values), max(values)]
        return None


def find_plotfile_dirs(path: Path) -> List[Path]:
    """Find all plotfile directories in a given path."""
    plotfile_dirs = []
    
    if (path / "Header").exists():
        # This is a plotfile directory
        return [path]
    
    # Search for subdirectories containing Header
    for item in path.iterdir():
        if item.is_dir() and (item / "Header").exists():
            plotfile_dirs.append(item)
    
    return plotfile_dirs


def main():
    parser = argparse.ArgumentParser(
        description='Parse AMReX plotfile directories and generate YAML descriptions.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Parse a single plotfile
  %(prog)s ocean_out/plt01080
  
  # Parse all plotfiles in a directory
  %(prog)s ocean_out/
  
  # Parse specific plotfiles
  %(prog)s ocean_out/plt00360 ocean_out/plt00720
  
  # Output to file
  %(prog)s ocean_out/ -o description.yaml
        """
    )
    
    parser.add_argument('paths', nargs='+', help='Plotfile directory/directories or parent directory')
    parser.add_argument('-o', '--output', help='Output YAML file (default: stdout)')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--no-blend', action='store_true', 
                       help='Do not blend multiple plotfiles (output each separately)')
    
    args = parser.parse_args()
    
    # Collect all plotfile directories
    all_plotfile_dirs = []
    for path_str in args.paths:
        path = Path(path_str)
        if not path.exists():
            print(f"Error: Path does not exist: {path}", file=sys.stderr)
            sys.exit(1)
        
        plotfile_dirs = find_plotfile_dirs(path)
        if not plotfile_dirs:
            print(f"Error: No plotfile directories found in: {path}", file=sys.stderr)
            sys.exit(1)
        
        all_plotfile_dirs.extend(plotfile_dirs)
    
    if args.verbose:
        print(f"Found {len(all_plotfile_dirs)} plotfile(s):", file=sys.stderr)
        for pf in all_plotfile_dirs:
            print(f"  {pf}", file=sys.stderr)
    
    # Parse and output
    try:
        if len(all_plotfile_dirs) == 1 or args.no_blend:
            # Single plotfile or no blending requested
            for plotfile_dir in all_plotfile_dirs:
                parser = AMReXPlotfileParser(plotfile_dir)
                description = parser.parse()
                
                if args.output:
                    # If multiple files and output specified, modify filename
                    if len(all_plotfile_dirs) > 1:
                        base = Path(args.output).stem
                        ext = Path(args.output).suffix
                        output_file = f"{base}_{plotfile_dir.name}{ext}"
                    else:
                        output_file = args.output
                    
                    with open(output_file, 'w') as f:
                        if HAS_YAML:
                            yaml.dump(description, f, default_flow_style=False, sort_keys=False)
                        else:
                            json.dump(description, f, indent=2)
                    
                    if args.verbose:
                        print(f"Wrote {output_file}", file=sys.stderr)
                else:
                    if HAS_YAML:
                        yaml.dump(description, sys.stdout, default_flow_style=False, sort_keys=False)
                    else:
                        json.dump(description, sys.stdout, indent=2)
                    if len(all_plotfile_dirs) > 1:
                        print("\n---\n")  # Separator between documents
        else:
            # Blend multiple plotfiles
            blender = AMReXPlotfileBlender()
            for plotfile_dir in all_plotfile_dirs:
                blender.add_plotfile(plotfile_dir)
            
            blended_description = blender.blend()
            
            if args.output:
                with open(args.output, 'w') as f:
                    if HAS_YAML:
                        yaml.dump(blended_description, f, default_flow_style=False, sort_keys=False)
                    else:
                        json.dump(blended_description, f, indent=2)
                
                if args.verbose:
                    print(f"Wrote {args.output}", file=sys.stderr)
            else:
                if HAS_YAML:
                    yaml.dump(blended_description, sys.stdout, default_flow_style=False, sort_keys=False)
                else:
                    json.dump(blended_description, sys.stdout, indent=2)
    
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
