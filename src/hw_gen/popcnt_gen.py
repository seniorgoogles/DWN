import os

class PopCountGenerator:
    
    @staticmethod
    def generate(num_inputs, output_dir, generate_figures=0, signed_in=0, msb_in=0, compression="optimalMinStages"):
        # Run shell command to generate the popcount module
        command = f"flopoco verbose=4 useTargetOpt=1 compression={compression} generateFigures={generate_figures} FixMultiAdder signedIn={signed_in} n={num_inputs} msbIn={msb_in} outputFile='popcnt.vhdl'"
        
        # Change directory to the output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Get current directory
        current_dir = os.getcwd()
        
        os.chdir(output_dir)
        
        # Execute the command
        print(f"Executing command: {command}")
     
        # Execute the command
        os.system(command)
        
        # Check if the command was successful
        if os.system(command) != 0:
            raise RuntimeError("Failed to generate the popcount module.")
        
        # Change back to the original directory
        os.chdir(current_dir)