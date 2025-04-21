import os

class LutLayerGenerator:
    @staticmethod
    def generate(lut_n, lut_count, lut_data_list, output_dir):
        # Create the VHDL code for the LutLayer module
        # Calculate widths
        data_in_width = lut_n * lut_count
        lut_entry_width = 2 ** lut_n

        # Build the INIT data array, reversing the bit order
        init_entries = [f'"{str(data[::-1])}"' for data in lut_data_list]
        init_data_str = ",\n        ".join(init_entries)

        # Build the port map dynamically based on lut_n
        port_map_lines = []
        for bit in range(lut_n):
            port_map_lines.append(f"                I{bit} => data_in((i*{lut_n}) + {bit}),")
        # Output mapping (no trailing comma)
        port_map_lines.append(f"                O  => data_out({lut_count - 1} - i)")
        port_map_str = "\n".join(port_map_lines)

        # VHDL module template
        vhdl_code = f"""
-- VHDL code for LutLayer module
-- This module takes a {lut_n}-bit input per LUT and produces a {lut_count}-bit output
-- The output is generated using {lut_count} LUTs of size {lut_n}

library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_arith.all;
use ieee.std_logic_unsigned.all;
library UNISIM;
use unisim.vcomponents.all;

entity LUTLayer is
    port(
        data_in  : in  std_logic_vector({data_in_width - 1} downto 0);
        data_out : out std_logic_vector({lut_count - 1} downto 0)
    );
end entity LUTLayer;

architecture Behavioral of LUTLayer is
    type lutlayer_data is array (natural range <>) of bit_vector({lut_entry_width - 1} downto 0);
    constant LUT_DATA : lutlayer_data := (
        {init_data_str}
    );
begin
    gen_lut: for i in 0 to {lut_count - 1} generate
        inst_lut: LUT{lut_n}
            generic map(
                INIT => LUT_DATA(i)
            )
            port map(
{port_map_str}
            );
    end generate gen_lut;
end Behavioral;
"""

        # Write the VHDL code to a file
        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(output_dir, "lutlayer.vhdl")
        with open(file_path, "w") as f:
            f.write(vhdl_code)

        print(f"LutLayer module generated in {file_path}")
