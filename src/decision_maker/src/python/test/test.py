import re
import uuid
shared_variable = "Hello, World!"
shared_variable2 = "Hello, World!111"


# Generate a unique number (UUID)
unique_number = uuid.uuid4()

# Shorten UUID to the first 7 letters
shortened_number = str(unique_number)[:7]
print(shortened_number)


# input_string = "[ 3.9701053  -1.67648514],[-2.24644262 -0.87444116],[ 0.35000026 -3.89999987],[5.50000034 2.70000022],[3.60000031 0.9000002 ]"

# # Replace whitespace with commas
# input_string = input_string.replace(" ", ",")

# # Remove consecutive commas
# input_string = re.sub(r',{2,}', ',', input_string)

# # Remove first and last comma of each bracket
# input_string = re.sub(r'\[(,.*?)(,)\]', r'[\1]', input_string)

# # Remove first comma in each bracket
# output_string = re.sub(r'\[,(.*?)]', r'[\1]', input_string)

# print(output_string)

# # Convert string to list
# output_list = eval(output_string)


# print(output_list)


# centroid_str = '[array([ 1.73647123, -1.0440681 ]), array([3.02500005, 1.45000002]), array([1.65000003, 2.55000004]), array([3.30000005, 4.50000006])]'

# # Remove the word "array" from the string
# centroid_str = centroid_str.replace('array', '')
# # Remove all parentheses from the string
# centroid_str = centroid_str.replace('(', '').replace(')', '')
# # Remove the first and last brackets from the string
# centroid_str = centroid_str[1:-1]

# # Remove all empty spaces in the string
# centroid_str = centroid_str.replace(" ", "")
# print(centroid_str)



# # Remove the surrounding brackets from the string
# centroid_str = centroid_str.strip('[]')

# # Split the string into individual coordinate pairs
# centroid_str = centroid_str.split('],[')

# # Process each pair to create the list of lists
# centroid_str = [list(map(float, pair.split(','))) for pair in centroid_str]

# print(centroid_str)





