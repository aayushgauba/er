import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

# Path to your notebook file
notebook_filename = "SE3_Mar8.ipynb"

# Read the notebook
with open(notebook_filename) as f:
    nb = nbformat.read(f, as_version=4)

# Create an ExecutePreprocessor instance
ep = ExecutePreprocessor(timeout=600, kernel_name='python3.8')

# Execute the notebook
ep.preprocess(nb, {'metadata': {'path': './'}})

# Open a text file for writing the outputs
with open("notebook_output.txt", "w", encoding="utf-8") as out_file:
    # Iterate over each cell and write its source and outputs
    for i, cell in enumerate(nb.cells):
        if cell.cell_type == 'code':
            out_file.write(f"\n\n--- Cell {i} ---\n")
            out_file.write("Code:\n")
            out_file.write(cell.source + "\n")
            out_file.write("Outputs:\n")
            for output in cell.get("outputs", []):
                if output.output_type == "stream":
                    out_file.write(output.get("text", "") + "\n")
                elif output.output_type == "execute_result":
                    data = output.get("data", {})
                    out_file.write(data.get("text/plain", "") + "\n")
                elif output.output_type == "error":
                    out_file.write("Error:\n")
                    out_file.write("\n".join(output.get("traceback", [])) + "\n")

print("Notebook output saved to notebook_output.txt")