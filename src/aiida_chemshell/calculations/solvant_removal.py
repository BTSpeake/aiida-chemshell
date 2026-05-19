"""Module for performing a solvant removal CalcJob."""

from aiida.common import CalcInfo, CodeInfo
from aiida.common.folders import Folder
from aiida.engine import CalcJob, CalcJobProcessSpec
from aiida.orm import Int, SinglefileData

from aiida_chemshell.utils import xyz_file_validator


class SolvantRemovalCalcJob(CalcJob):
    """CalcJob to extract a solvant from a complex for NEB calculations."""

    _SCRIPT_NAME = "separate_solvant.py"

    @classmethod
    def define(cls, spec: CalcJobProcessSpec) -> None:
        """Define the CalcJob spec for solvant removal."""
        super().define(spec)
        spec.input(
            "structure",
            valid_type=SinglefileData,
            required=True,
            validator=xyz_file_validator,
            help=(
                "Input structure file containing ligand+solvant gas phase complex "
                "in XYZ format."
            ),
        )
        spec.input(
            "num_ligand_atoms",
            valid_type=Int,
            required=True,
            help="Number of atoms within the core ligand molecule.",
        )
        spec.output(
            "unbound_structure",
            valid_type=SinglefileData,
            required=True,
            help="Gas phase structure for the unboun ligand+solvant complex.",
        )

        spec.inputs["metadata"]["options"]["resources"].default = {
            "num_machines": 1,
            "num_mpiprocs_per_machine": 1,
        }
        spec.inputs["metadata"]["options"]["parser_name"].default = "chemshell.unbind"

    def prepare_for_submission(self, folder: Folder) -> CalcInfo:
        """Prepare the CalcJob for submission."""
        py_scipt = self._generate_python_script()
        with folder.open(SolvantRemovalCalcJob._SCRIPT_NAME, "w") as handle:
            handle.write(py_scipt)

        code_info = CodeInfo()
        code_info.code_uuid = self.inputs.code.uuid
        code_info.cmdline_params = [SolvantRemovalCalcJob._SCRIPT_NAME]

        calc_info = CalcInfo()
        calc_info.codes_info = [code_info]
        calc_info.retrieve_temporary_list = []
        calc_info.provenance_exclude_list = []
        calc_info.retrieve_list = [self._generate_output_filename()]
        calc_info.local_copy_list = [
            (
                self.inputs.structure.uuid,
                self.inputs.structure.filename,
                self.inputs.structure.filename,
            ),
        ]

        return calc_info

    @classmethod
    def generate_output_filename(cls, input_filename) -> str:
        """Generate a filename based on the name of the initial structure file."""
        return input_filename.replace(".xyz", "_unbound.xyz")

    def _generate_output_filename(self) -> str:
        """Generate the output filename."""
        return self.generate_output_filename(self.inputs.structure.filename)

    def _generate_python_script(self) -> str:
        """Generate the Python script to be executed."""
        return f"""
import numpy
with open("{self.inputs.structure.filename}", "r") as f:
    lines = f.readlines()
natoms = int(lines[0].strip())
ligand = []
for i in range(2, {int(self.inputs.num_ligand_atoms)} + 2):
    line = lines[i].split()
    ligand.append(numpy.array([float(line[1]), float(line[2]), float(line[3])]))
solvant = []
for i in range({int(self.inputs.num_ligand_atoms)} + 2, natoms + 2):
    line = lines[i].split()
    solvant.append(numpy.array([float(line[1]), float(line[2]), float(line[3])]))
ligand_center = numpy.mean(ligand, axis=0)
solvant_center = numpy.mean(solvant, axis=0)
vector = solvant_center - ligand_center
distance = numpy.linalg.norm(vector)
vector /= distance
displacement = vector * 10.0  # Move solvant 10 Angstroms away
with open("{self._generate_output_filename()}", "w") as f:
    f.write(f"{{natoms}}\\n")
    f.write("Unbound ligand+solvant complex\\n")
    for i in range(2, {int(self.inputs.num_ligand_atoms)} + 2):
        f.write(lines[i])
    for i in range({int(self.inputs.num_ligand_atoms)} + 2, natoms + 2):
        line = lines[i].split()
        x = float(line[1]) + displacement[0]
        y = float(line[2]) + displacement[1]
        z = float(line[3]) + displacement[2]
        f.write(f"{{line[0]}} {{x:.8f}} {{y:.8f}} {{z:.8f}}\\n")
"""
