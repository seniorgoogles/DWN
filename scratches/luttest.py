class LookupTable():
    def __init__(self, luts_data, num_inputs):
        self.num_inputs = num_inputs
        self.luts_data = luts_data

        self.luts_output = dict()
        
        #print(f"\n{luts_data=}")
        
        for i in range(2**num_inputs):
            self.luts_output[f"{i:0{num_inputs}b}"] = luts_data[i]
            
        
        #print(f"{self.luts_output=}")
        
    def __call__(self, x):
        return self.luts_output[x]
    
    
"""
lut = LookupTable("0010", 2)
print(lut("00"))
print(lut("01"))
print(lut("10"))
print(lut("11"))

lut = LookupTable("0001", 2)
print(lut("00"))
print(lut("01"))
print(lut("10"))
print(lut("11"))

lut = LookupTable("0011", 2)
print(lut("00"))
print(lut("01"))
print(lut("10"))
print(lut("11"))

lut = LookupTable("0010", 2)
print(lut("00"))
print(lut("01"))
print(lut("10"))
print(lut("11"))

lut = LookupTable("1100", 2)
print(lut("00"))
print(lut("01"))
print(lut("10"))
print(lut("11"))
"""

lut_data = ["0010", "0001", "0011", "0010", "1100"]
luts = []
for i in range(len(lut_data)):
    luts.append(LookupTable(lut_data[i], 2))
    
input_data = [
    "1011110111",
    "1011110111",
    "0010100000",
    "0010100000",
    "1111100111",
    "0010100000",
    "0101100011",
    "0001100011",
    "1011100111",
    "0110100000",
]

print()

for i in range(len(input_data)):
    out_str = ""

    for j in range(len(luts)):
        #print(luts[j](input_data[(j*2):(j*2)+1]), end="")
        out_str += luts[j](input_data[i][j*2:j*2+2])[::-1]
        
    print("{0} -> {1}".format(input_data[i], out_str[::-1]))