import numpy as np
import torch
import os
import sys
from sklearn.model_selection import train_test_split
import openml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from torch_dwn.binarization import DistributiveThermometer

class ThermometerGen:
    @staticmethod
    def generate(bins: int, scale_factor: int, dataset_str: str, output_dir: str) -> str:
        if bins < 1:
            raise ValueError("Number of bins must be at least 1.")

        # Load and process dataset
        if "jsc" in dataset_str:
            dataset = openml.datasets.get_dataset(42468)
            df_features, df_labels, _, _ = dataset.get_data(
                dataset_format='dataframe', target=dataset.default_target_attribute
            )
            features = df_features.values.astype(np.float32)
            label_names = list(df_labels.unique())
            labels = np.array(df_labels.map(lambda x: label_names.index(x)).values)

        # Train/test split
        x_train, x_test, y_train, y_test = train_test_split(
            features, labels, train_size=0.8, random_state=42
        )

        # Flatten data
        x_train = x_train.flatten()

        # Fit thermometer encoder
        thermometer = DistributiveThermometer(bins, feature_wise=True).fit(torch.tensor(x_train))

        # Scale thresholds and convert to integer
        thermometer.thresholds = (thermometer.thresholds * scale_factor).to(torch.int32).tolist()

        # if thresholds are not unique, raise an error
        if len(thermometer.thresholds) != len(set(thermometer.thresholds)):
            raise ValueError("Thresholds are not unique.")
        
        # Create VHDL string
        vhdl_lines = [
            "library ieee;",
            "use ieee.std_logic_1164.all;",
            "use ieee.numeric_std.all;",
            "",
            "entity thermometer is",
            "    port (",
            "        clk      : in std_logic;",
            "        rst      : in std_logic;",
            "        start    : in std_logic;",
            "        data_in  : in std_logic_vector(31 downto 0);",
            f"        data_out : out std_logic_vector({bins - 1} downto 0)",
            "    );",
            "end entity;",
            "",
            "architecture behavior of thermometer is",
            f"    signal data_out_int : std_logic_vector({bins - 1} downto 0);",
            "    signal data_in_int  : signed(31 downto 0);",
            "begin",
            "    process(clk, rst)",
            "    begin",
            "        if rst = '1' then",
            "            data_out_int <= (others => '0');",
            "        elsif rising_edge(clk) then",
            "            if start = '1' then",
            "                data_in_int <= signed(data_in);",
            "                -- Thermometer encoding (cumulative style)"
        ]

        # Build cumulative binary thermometer strings
        for i, threshold in enumerate(thresholds):
            thermometer_code = "1" * i + "0" * (bins - i)
            if_line = "if" if i == 0 else "elsif"
            vhdl_lines.append(
                f"                {if_line} data_in_int < to_signed({threshold}, 32) then"
            )
            vhdl_lines.append(
                f"                    data_out_int <= \"{thermometer_code}\";"
            )

        # Final 'else' case (all bits high)
        vhdl_lines.append("                else")
        vhdl_lines.append(f"                    data_out_int <= \"{'1'*bins}\";")
        vhdl_lines.append("                end if;")
        vhdl_lines.append("            end if;")
        vhdl_lines.append("        end if;")
        vhdl_lines.append("    end process;")
        vhdl_lines.append("    data_out <= data_out_int;")
        vhdl_lines.append("end architecture;")

        # Join VHDL lines
        vhdl_str = "\n".join(vhdl_lines)

        # Save VHDL file
        os.makedirs(output_dir, exist_ok=True)
        vhdl_file_path = os.path.join(output_dir, "thermometer.vhdl")
        with open(vhdl_file_path, "w") as f:
            f.write(vhdl_str)
            
    
if __name__ == "__main__":
    # Example usage
    bins = 200
    dataset_str = "jsc"
    output_dir = "./output/thermometer"
    scale_factor = 100000
    
    generator = ThermometerGen()
    generator.generate(bins, scale_factor, dataset_str, output_dir)