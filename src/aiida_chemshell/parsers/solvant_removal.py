"""Module for parsing a solvant removcal CalcJob."""

from aiida.engine import ExitCode
from aiida.orm import SinglefileData
from aiida.parsers.parser import Parser

from aiida_chemshell.calculations.solvant_removal import SolvantRemovalCalcJob


class UnbindJobParser(Parser):
    """AiiDA parser plugin for unbinding molecules job."""

    def parse(self, **kwargs) -> ExitCode:
        """Parser the ChemShell utility job."""
        fname = SolvantRemovalCalcJob.generate_output_filename(
            self.node.inputs.structure.filename
        )

        with self.retrieved.open(fname, "r") as f:
            self.out(
                "unbound_structure",
                SinglefileData(
                    file=f, filename=fname, label="Unbound Structure", description=""
                ),
            )

        return ExitCode(0)
