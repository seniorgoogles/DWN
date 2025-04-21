import os 
import sys


class GroupSumGenerator:
    
    @staticmethod
    def generate(population_width, num_classes, output_dir):
        # Create the VHDL code for the GroupSum module
        vhdl_code = f"""
-- VHDL code for GroupSum module
-- This module takes a {population_width * num_classes}-bit input and produces a {num_classes}-bit output
-- The output indicates which class has the maximum number of ones in the input
-- The input is divided into {num_classes} classes, each containing {population_width} bits
-- The output is a {num_classes}-bit vector, where the bit corresponding to the class with the maximum number of ones is set to 1

library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_arith.all;
use ieee.std_logic_unsigned.all;

entity GroupSum is
    port(
        data_in : in std_logic_vector({(population_width * num_classes) - 1} downto 0);
        data_out : out std_logic_vector({num_classes - 1} downto 0)    
    );
end entity GroupSum;

architecture Behavioral of GroupSum is
begin
    p_groupsum: process(data_in)
        variable index_max_value : integer := 0;
        variable max_value : integer := 0;
        variable curr_value : integer := 0;
        variable index : integer := 0;

        begin
            -- Iterate over the classes
            for i in 0 to {num_classes - 1} loop

                -- Count the number of ones in the class
                for j in 0 to {population_width - 1} loop
                    index := j + {population_width} * i;
                    if data_in(index) = '1' then
                        curr_value := curr_value + 1 * (2 ** j);
                        report("curr_value: " & integer'image(curr_value));
                    end if;
                end loop;

                -- Check if the current class has more ones than the previous max
                if curr_value > max_value then
                    max_value := curr_value;
                    index_max_value := i;
                end if;

                --report("Class " & integer'image(i) & " has " & integer'image(curr_value));
                curr_value := 0;

            end loop;
            
            --report("Max value: " & integer'image(max_value) & " at Index: " & integer'image(index_max_value));
            data_out <= (others => '0');
            data_out(index_max_value) <= '1';
            index_max_value := 0;
            max_value := 0;
            curr_value := 0;
    end process;
end architecture Behavioral;
"""
        # Write the VHDL code to a file
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "groupsum.vhdl"), "w") as f:
            f.write(vhdl_code)
        
        print(f"GroupSum module generated in {output_dir}/groupsum.vhdl")