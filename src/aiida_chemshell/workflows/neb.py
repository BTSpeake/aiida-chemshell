"""Module to define a NEB based workflow."""

from aiida.engine import ToContext, WorkChain
from aiida.orm import ArrayData, Code, Int, SinglefileData, StructureData
from aiida.plugins.factories import CalculationFactory

from aiida_chemshell.calculations.solvant_removal import SolvantRemovalCalcJob

ChemShellCalculation = CalculationFactory("chemshell")


class CLFUNudgedElasticBandWorkChain(WorkChain):
    """WorkChain to run a CLF Ultra NEB calculation using Chemshell."""

    @classmethod
    def define(cls, spec):
        """Define the input/output specifications for a NEB workflow."""
        super().define(spec)

        spec.input(
            "chemshell",
            valid_type=Code,
            required=True,
            help="The ChemShell AiiDA code instance to use for the WorkChain.",
        )

        spec.input(
            "initial_structure",
            valid_type=(SinglefileData, StructureData),
            required=True,
            help="Initial structure for the NEB calculation.",
        )
        spec.input(
            "num_ligand_atoms",
            valid_type=Int,
            required=True,
            help="Number of atoms within the core ligang molecule.",
        )
        spec.input(
            "num_images",
            valid_type=Int,
            default=Int(10),
            help="Number of images for the NEB calculation.",
        )
        spec.outline(
            cls.determine_final_state,
            cls.run_neb,
            cls.finalize,
        )
        spec.output("neb_path", valid_type=ArrayData, help="The computed NEB path.")

    def determine_final_state(self):
        """Determine the geometry gor the gas phase seperated molecules."""
        self.report("Extracting the unbound ligand/solvant system")
        inputs = {
            "code": self.inputs.chemshell,
            "structure": self.inputs.initial_structure,
            "num_ligand_atoms": self.inputs.num_ligand_atoms,
        }
        future = self.submit(SolvantRemovalCalcJob, **inputs)
        return ToContext(final_state=future)

    def setup(self):
        """Set up the NEB calculation."""
        self.report(f"Setting up NEB calculation with {self.inputs.num_images} images.")
        # Additional setup code here

    def run_neb(self):
        """Run the NEB calculation."""
        self.report("Running NEB calculation...")
        # Code to execute the NEB calculation using Chemshell
        inputs = {
            "structure": self.inputs.initial_structure,
            "structure2": self.ctx.final_state.outputs.unbound_structure,
            "qm_parameters": {
                "theory": "PySCF",
                "method": "hf",
                "basis": "6-31G",
                "functional": "b3lyp",
            },
            "optimisation_parameters": {
                "neb": "frozen",
            },
            "code": self.inputs.chemshell,
        }
        future = self.submit(ChemShellCalculation, **inputs)
        return ToContext(neb=future)

    def finalize(self):
        """Finalize and store results."""
        self.report("Finalizing NEB calculation and storing results.")
        # Code to collect and store the results in neb_path output
