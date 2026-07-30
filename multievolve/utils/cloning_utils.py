# updated 7/30/2026: require at least three SNVs per amino-acid change; support '*' stop codons
from concurrent.futures import ProcessPoolExecutor
import copy
import re

import pandas as pd
import numpy as np
import os

from Bio import Align, SeqIO
from Bio.Data import CodonTable
from Bio.Seq import Seq
from Bio.SeqUtils import MeltingTemp as mt
from Bio.SeqRecord import SeqRecord
from typing import Optional, Tuple

codon_dicts = {
    'human': {
    'F': 'TTT', 'L': 'CTG', 'Y': 'TAT', 'H': 'CAT', 'Q': 'CAG',
    'I': 'ATT', 'M': 'ATG', 'N': 'AAT', 'K': 'AAG', 'V': 'GTG',
    'D': 'GAT', 'E': 'GAG', 'S': 'TCT', 'C': 'TGT', 'W': 'TGG',
    'P': 'CCT', 'R': 'CGG', 'T': 'ACT', 'A': 'GCT', 'G': 'GGG',
    '*': 'TGA',
    },
    'ecoli': {
        'F': 'TTT', 'L': 'CTG', 'Y': 'TAT', 'H': 'CAT', 'Q': 'CAG',
        'I': 'ATT', 'M': 'ATG', 'N': 'AAC', 'K': 'AAA', 'V': 'GTG',
        'D': 'GAT', 'E': 'GAA', 'S': 'TCT', 'C': 'TGC', 'W': 'TGG',
        'P': 'CCG', 'R': 'CGT', 'T': 'ACC', 'A': 'GCG', 'G': 'GGC',
        '*': 'TAA',
    },
    'yeast': {
        'F': 'TTT', 'L': 'CTA', 'Y': 'TAT', 'H': 'CAT', 'Q': 'CAA',
        'I': 'ATT', 'M': 'ATG', 'N': 'AAT', 'K': 'AAA', 'V': 'GTT',
        'D': 'GAT', 'E': 'GAA', 'S': 'TCT', 'C': 'TGT', 'W': 'TGG',
        'P': 'CCA', 'R': 'AGA', 'T': 'ACT', 'A': 'GCT', 'G': 'GGT',
        '*': 'TAA',
    }
}
class MultiAssemblyDesigner:

    """Designs oligos for protein mutations.

    Args:
        data (pd.DataFrame): DataFrame containing mutation data.
        start_seq_fasta (str): Path to FASTA file with starting sequence.
        overhang (int): Overhang length.
        species (str): Species, 'human', 'ecoli', or 'yeast'.
        oligo_direction (str): Direction of oligo, 'bottom' or 'top'.
        tm (float): Target melting temperature.
        output (str): Type of output, 'design' or 'update'.
        min_snvs_per_mutation (int): Minimum nucleotide substitutions used to
            encode each amino-acid substitution, including nonsense mutations. If the missense codon cannot provide
            enough substitutions, nearby synonymous marker substitutions are added.
        silent_marker_radius_codons (int): Maximum distance from the missense
            codon to search for synonymous marker codons.
    """

    def __init__(
        self,
        data,
        start_seq_fasta,
        overhang,
        species='human',
        oligo_direction='bottom',
        tm=80,
        output='design',
        min_snvs_per_mutation=3,
        silent_marker_radius_codons=5,
    ):
        print("Initializing MultiAssemblyDesigner...")
        self.data = data.rename(columns={data.columns[0]: 'aa_mut'})
        self.data['aa_mut'] = self.data['aa_mut'].apply(self._sort_mutations)
        self.fasta_dir = os.path.dirname(start_seq_fasta)
        print(f'The melting temperature is {tm}')
        self.tm = tm
        self.start_seq = SeqIO.read(start_seq_fasta, "fasta").seq.upper()
        self.overhang = int(overhang)
        self.oligo_direction = oligo_direction
        self.codon_dict = codon_dicts[species]
        self.min_snvs_per_mutation = int(min_snvs_per_mutation)
        self.silent_marker_radius_codons = int(silent_marker_radius_codons)

        if self.min_snvs_per_mutation < 1:
            raise ValueError("min_snvs_per_mutation must be at least 1")
        if self.silent_marker_radius_codons < 0:
            raise ValueError("silent_marker_radius_codons must be non-negative")

        self.codons_by_aa = self._build_codons_by_aa()
        self._process_mutations()
        self._design_oligos()
        self._find_unique_mutant_oligos()

        if output == 'design':
            print('Exporting design...')
            self._export_design()
        elif output == 'update':
            print('Updating oligo IDs...')
            self._modify_oligo_id()

    def _sort_mutations(self, mutation_string):
        """Sort mutations within a slash-delimited string by amino-acid position."""
        mutations = mutation_string.split('/')
        return sorted(mutations, key=lambda x: int(''.join(filter(str.isdigit, x))))

    def _process_mutations(self):
        """Resolve each amino-acid change into its complete nucleotide edit block."""
        self.data[[
            'Positions',
            'Reference_bases',
            'Alternative_bases',
            'Nucleotide_edits',
        ]] = self.data.apply(
            lambda x: pd.Series(
                self._get_codon_mutation_list(
                    x['aa_mut'],
                    self.codon_dict,
                    self.overhang,
                    str(self.start_seq),
                )
            ),
            axis=1,
        )

        self.data['mut_seq'] = self.data['Nucleotide_edits'].apply(self._get_mut_seq)
        self.data['SNVs_per_mutation'] = self.data['Nucleotide_edits'].apply(
            lambda blocks: [
                sum(
                    self._snv_distance(edit['reference'], edit['alternate'])
                    for edit in block
                )
                for block in blocks
            ]
        )

    def _design_oligos(self):
        """Design oligos for each mutation row."""
        self.data[['oligos', 'oligo_mut']] = self.data.apply(
            lambda x: pd.Series(self._design_oligo_pipeline(x)),
            axis=1,
        )

    @staticmethod
    def _build_codons_by_aa():
        """Return all standard DNA codons grouped by encoded amino acid."""
        codons_by_aa = {}
        standard = CodonTable.unambiguous_dna_by_name['Standard']
        for codon, aa in standard.forward_table.items():
            codons_by_aa.setdefault(aa, []).append(codon)
        # Biopython's forward_table excludes termination codons, so add them
        # explicitly. This permits mutation names such as Q123*.
        codons_by_aa['*'] = list(standard.stop_codons)
        return {aa: sorted(codons) for aa, codons in codons_by_aa.items()}

    @staticmethod
    def _snv_distance(sequence_a, sequence_b):
        """Count substitutions between two equal-length nucleotide strings."""
        if len(sequence_a) != len(sequence_b):
            raise ValueError("SNV distance requires equal-length sequences")
        return sum(
            a != b
            for a, b in zip(sequence_a.upper(), sequence_b.upper())
        )

    def _choose_missense_codon(self, wt_codon, target_aa):
        """Choose a target codon with maximal separation from the WT codon.

        The target amino acid may be '*' to encode a termination codon.
        """
        if target_aa not in self.codons_by_aa:
            raise ValueError(
                f"Unsupported target amino-acid symbol {target_aa!r}; "
                "use a standard one-letter code or '*' for stop."
            )
        preferred = self.codon_dict[target_aa]
        if self._snv_distance(wt_codon, preferred) >= self.min_snvs_per_mutation:
            return preferred

        candidates = self.codons_by_aa[target_aa]
        return min(
            candidates,
            key=lambda codon: (
                -self._snv_distance(wt_codon, codon),
                codon != preferred,
                codon,
            ),
        )

    def _choose_silent_codon(self, wt_codon, snvs_needed):
        """Choose a synonymous codon that contributes marker substitutions."""
        aa = str(Seq(wt_codon).translate())
        preferred = self.codon_dict[aa]
        candidates = [
            codon
            for codon in self.codons_by_aa[aa]
            if codon != wt_codon
        ]
        if not candidates:
            return None

        sufficient = [
            codon
            for codon in candidates
            if self._snv_distance(wt_codon, codon) >= snvs_needed
        ]
        if sufficient:
            target_distance = min(
                self._snv_distance(wt_codon, codon)
                for codon in sufficient
            )
            candidates = [
                codon
                for codon in sufficient
                if self._snv_distance(wt_codon, codon) == target_distance
            ]
        else:
            target_distance = max(
                self._snv_distance(wt_codon, codon)
                for codon in candidates
            )
            candidates = [
                codon
                for codon in candidates
                if self._snv_distance(wt_codon, codon) == target_distance
            ]

        return min(candidates, key=lambda codon: (codon != preferred, codon))

    def _get_codon_mutation_list(self, mut_ls, codon_dict, overhang, start_seq):
        """Return target codons plus complete missense-specific edit blocks."""
        del codon_dict  # retained in the signature for backward compatibility

        cds_nt_length = len(start_seq) - (2 * overhang)
        if cds_nt_length <= 0 or cds_nt_length % 3 != 0:
            raise ValueError(
                "The FASTA sequence minus both overhangs must contain a non-empty "
                "CDS whose length is divisible by three"
            )

        protein_length = cds_nt_length // 3
        target_aa_positions = {int(mut[1:-1]) for mut in mut_ls}
        reserved_marker_positions = set()

        positions = []
        reference_codons = []
        alternative_codons = []
        edit_blocks = []

        for mut in mut_ls:
            aa_position = int(mut[1:-1])
            if aa_position < 1 or aa_position > protein_length:
                raise ValueError(f"Mutation {mut} is outside the encoded protein")

            cds_position = aa_position * 3 - 2
            full_sequence_position = cds_position + overhang
            target_start0 = full_sequence_position - 1
            old_codon = start_seq[target_start0:target_start0 + 3]
            observed_aa = str(Seq(old_codon).translate())

            if observed_aa != mut[0]:
                raise ValueError(
                    f"{mut} is not a true mutation from "
                    f"{observed_aa}{aa_position}"
                )
            if observed_aa == mut[-1]:
                raise ValueError(f"{mut} is synonymous rather than missense")

            new_codon = self._choose_missense_codon(old_codon, mut[-1])
            edits = [{
                'position': full_sequence_position,
                'reference': old_codon,
                'alternate': new_codon,
                'kind': 'nonsense' if mut[-1] == '*' else 'missense',
            }]
            snv_count = self._snv_distance(old_codon, new_codon)

            for radius in range(1, self.silent_marker_radius_codons + 1):
                if snv_count >= self.min_snvs_per_mutation:
                    break

                for delta in (-radius, radius):
                    marker_aa_position = aa_position + delta
                    if not 1 <= marker_aa_position <= protein_length:
                        continue
                    if marker_aa_position in target_aa_positions:
                        continue
                    if marker_aa_position in reserved_marker_positions:
                        continue

                    marker_start0 = overhang + (marker_aa_position - 1) * 3
                    marker_wt_codon = start_seq[marker_start0:marker_start0 + 3]
                    marker_alt_codon = self._choose_silent_codon(
                        marker_wt_codon,
                        self.min_snvs_per_mutation - snv_count,
                    )
                    if marker_alt_codon is None:
                        # Methionine and tryptophan have no synonymous codons.
                        continue

                    edits.append({
                        'position': marker_start0 + 1,
                        'reference': marker_wt_codon,
                        'alternate': marker_alt_codon,
                        'kind': 'synonymous_marker',
                    })
                    reserved_marker_positions.add(marker_aa_position)
                    snv_count += self._snv_distance(
                        marker_wt_codon,
                        marker_alt_codon,
                    )
                    if snv_count >= self.min_snvs_per_mutation:
                        break

            if snv_count < self.min_snvs_per_mutation:
                raise ValueError(
                    f"Could not encode {mut} with at least "
                    f"{self.min_snvs_per_mutation} SNVs within +/-"
                    f"{self.silent_marker_radius_codons} codons. Increase "
                    "silent_marker_radius_codons or reduce min_snvs_per_mutation."
                )

            positions.append(full_sequence_position)
            reference_codons.append(old_codon)
            alternative_codons.append(new_codon)
            edit_blocks.append(edits)

        return positions, reference_codons, alternative_codons, edit_blocks

    def _flatten_edits(self, edit_blocks):
        """Flatten edit blocks, deduplicate identical edits, and reject conflicts."""
        unique_edits = {}
        for block in edit_blocks:
            for edit in block:
                key = (int(edit['position']), len(edit['reference']))
                if key in unique_edits:
                    previous = unique_edits[key]
                    if previous['alternate'].upper() != edit['alternate'].upper():
                        raise ValueError(
                            f"Conflicting nucleotide edits at position "
                            f"{edit['position']}"
                        )
                    continue
                unique_edits[key] = edit

        return sorted(
            unique_edits.values(),
            key=lambda edit: int(edit['position']),
        )

    def _apply_edits(self, seq, edits):
        """Apply equal-length nucleotide substitutions to a sequence."""
        edited_seq = Seq(str(seq))
        for edit in sorted(edits, key=lambda item: int(item['position'])):
            start0 = int(edit['position']) - 1
            reference = edit['reference'].upper()
            observed = str(
                edited_seq[start0:start0 + len(reference)]
            ).upper()
            if observed != reference:
                raise ValueError(
                    f"Reference mismatch at nucleotide {edit['position']}: "
                    f"expected {reference}, observed {observed}"
                )
            edited_seq = (
                edited_seq[:start0]
                + edit['alternate'].lower()
                + edited_seq[start0 + len(reference):]
            )
        return edited_seq

    def _design_oligo_pipeline(self, row):
        """Design one or more oligos for a row of amino-acid changes."""
        pos_start_ls = []
        pos_end_ls = []
        for edit_block in row['Nucleotide_edits']:
            pos_start, pos_end = self._design_mutant_oligo(
                self.start_seq,
                edit_block,
                result='positions',
            )
            pos_start_ls.append(pos_start)
            pos_end_ls.append(pos_end)

        oligos = []
        oligo_mt_mapping = []
        i = 0
        while i < len(row['Positions']):
            mutation_names = [row['aa_mut'][i]]
            start_index = pos_start_ls[i]
            index_i = i

            if i < len(row['Positions']) - 1:
                n = 0
                while pos_end_ls[i + n] >= pos_start_ls[i + n + 1]:
                    n += 1
                    mutation_names.append(row['aa_mut'][i + n])
                    if i + n + 1 == len(row['Positions']):
                        break
                index_f = i + n
                i += n
                end_index = pos_end_ls[index_f]
            else:
                index_f = i
                end_index = pos_end_ls[i]

            oligos.append(
                str(
                    self._get_mutant_oligo_by_pos(
                        self.start_seq,
                        row['Nucleotide_edits'],
                        start_index,
                        end_index,
                        index_i,
                        index_f,
                    )
                )
            )
            oligo_mt_mapping.append("-".join(mutation_names))
            i += 1

        return oligos, oligo_mt_mapping

    def _get_mut_seq(self, edit_blocks):
        """Generate the complete variant sequence, including silent markers."""
        return str(
            self._apply_edits(
                copy.deepcopy(self.start_seq),
                self._flatten_edits(edit_blocks),
            )
        )

    def _design_mutant_oligo(self, seq, edit_block, result='oligo'):
        """Design an oligo containing a coding edit and its silent markers."""
        edits = self._flatten_edits([edit_block])
        mut_seq = self._apply_edits(seq, edits)
        wt_seq = Seq(str(seq))

        first_edit_start = min(int(edit['position']) - 1 for edit in edits)
        last_edit_end = max(
            int(edit['position']) - 1 + len(edit['reference'])
            for edit in edits
        )
        start_index = max(0, first_edit_start - 11)
        end_index = min(len(seq), last_edit_end + 11)

        if self.oligo_direction == 'bottom':
            oligo = mut_seq[start_index:end_index].reverse_complement()
            wt_oligo = wt_seq[start_index:end_index].reverse_complement()
        else:
            oligo = mut_seq[start_index:end_index]
            wt_oligo = wt_seq[start_index:end_index]

        while mt.Tm_NN(oligo, Na=50, K=25, Tris=35, Mg=10) <= self.tm:
            can_extend_left = start_index > 0
            can_extend_right = end_index < len(seq)
            if not can_extend_left and not can_extend_right:
                raise ValueError(
                    "Unable to reach the requested oligo Tm before the "
                    "sequence boundaries"
                )

            if (len(oligo) % 2 == 0 and can_extend_left) or not can_extend_right:
                start_index -= 1
            else:
                end_index += 1

            if self.oligo_direction == 'bottom':
                oligo = mut_seq[start_index:end_index].reverse_complement()
                wt_oligo = wt_seq[start_index:end_index].reverse_complement()
            else:
                oligo = mut_seq[start_index:end_index]
                wt_oligo = wt_seq[start_index:end_index]

        if result == 'oligo':
            return (
                str(oligo),
                str(wt_oligo),
                round(mt.Tm_NN(oligo, Na=50, K=25, Tris=35, Mg=10), 2),
            )
        return start_index + 1, end_index + 1

    def _get_mutant_oligo_by_pos(
        self,
        seq,
        edit_blocks,
        start,
        end,
        index_i,
        index_f,
    ):
        """Return a merged oligo for overlapping coding-edit blocks."""
        edits = self._flatten_edits(edit_blocks[index_i:index_f + 1])
        edited_seq = self._apply_edits(seq, edits)
        mod_start = int(start) - 1
        mod_end = int(end) - 1
        oligo = edited_seq[mod_start:mod_end]
        if self.oligo_direction == 'bottom':
            return oligo.reverse_complement()
        return oligo

    def _find_unique_mutant_oligos(self):
        """Identify unique oligos while retaining sequence-dependent barcodes."""
        oligos = [
            item
            for sublist in self.data['oligos'].tolist()
            for item in sublist
        ]
        oligo_mutation = [
            item
            for sublist in self.data['oligo_mut'].tolist()
            for item in sublist
        ]
        df = pd.DataFrame({
            'oligos': oligos,
            'mutation': oligo_mutation,
        }).drop_duplicates(subset=['mutation', 'oligos'], keep='first')
        df['oligo_id'] = range(len(df))

        oligo_dict = {
            (row.mutation, row.oligos): row.oligo_id
            for row in df.itertuples(index=False)
        }
        self.data['oligo_id'] = self.data.apply(
            lambda row: [
                oligo_dict[(mutation, oligo)]
                for mutation, oligo in zip(
                    row['oligo_mut'],
                    row['oligos'],
                )
            ],
            axis=1,
        )
        self.oligos = df
        self.data[['oligo_id', 'oligo_mut']] = self.data.apply(
            self._sort_oligos,
            axis=1,
        )

    def _sort_oligos(self, row):
        """Sort oligo IDs and corresponding mutation labels together."""
        paired_data = list(zip(row['oligo_id'], row['oligo_mut']))
        paired_data.sort(key=lambda x: x[0])
        sorted_ids, sorted_muts = map(list, zip(*paired_data))
        return pd.Series({
            'oligo_id': sorted_ids,
            'oligo_mut': sorted_muts,
        })

    def _export_df_with_lists(self, df, filepath, delimiter=','):
        """Export DataFrame list columns as delimiter-separated strings."""
        df_to_save = df.copy()
        for column in df_to_save.columns:
            if df_to_save[column].apply(lambda x: isinstance(x, list)).any():
                df_to_save[column] = df_to_save[column].apply(
                    lambda x: delimiter.join(str(item) for item in x)
                    if isinstance(x, list)
                    else x
                )
        df_to_save.to_csv(filepath, index=False)

    def _import_df_with_lists(self, filepath, delimiter=','):
        """Import a CSV and restore simple delimiter-separated list columns."""
        df = pd.read_csv(filepath)
        for column in df.columns:
            try:
                if df[column].dtype == 'object':
                    first_value = str(df[column].iloc[0])
                    if delimiter in first_value:
                        def convert_to_list(value):
                            if pd.isna(value):
                                return []
                            items = str(value).split(delimiter)
                            try:
                                return [
                                    float(item) if '.' in item else int(item)
                                    for item in items
                                ]
                            except ValueError:
                                return items
                        df[column] = df[column].apply(convert_to_list)
            except Exception:
                continue
        return df

    def _export_encoding_map(self):
        """Export the expected nucleotide barcode for every amino-acid change."""
        records = []
        for _, row in self.data.iterrows():
            variant = '/'.join(row['aa_mut'])
            for mutation, edit_block in zip(
                row['aa_mut'],
                row['Nucleotide_edits'],
            ):
                edit_strings = []
                for edit in edit_block:
                    cds_position = int(edit['position']) - self.overhang
                    edit_strings.append(
                        f"{edit['kind']}:{edit['reference']}@"
                        f"{cds_position}>{edit['alternate']}"
                    )
                records.append({
                    'variant': variant,
                    'mutation': mutation,
                    'snv_count': sum(
                        self._snv_distance(
                            edit['reference'],
                            edit['alternate'],
                        )
                        for edit in edit_block
                    ),
                    'nucleotide_edits': ';'.join(edit_strings),
                })

        pd.DataFrame(records).drop_duplicates().to_csv(
            os.path.join(self.fasta_dir, 'mutation_encoding.csv'),
            index=False,
        )

    def _export_design(self):
        """Export the cloning sheet, oligos, and nucleotide encoding map."""
        self._export_df_with_lists(
            self.data[['oligo_id', 'oligo_mut']].copy(),
            os.path.join(self.fasta_dir, 'cloning_sheet.csv'),
        )
        self.oligos.to_csv(
            os.path.join(self.fasta_dir, 'oligos.csv'),
            index=False,
        )
        self._export_encoding_map()

    def _modify_oligo_id(self):
        """Update cloning-sheet IDs from the current oligos file."""
        self.oligos = self._import_df_with_lists(
            os.path.join(self.fasta_dir, 'oligos.csv')
        )
        oligo_dict = {
            (row.mutation, row.oligos): row.oligo_id
            for row in self.oligos.itertuples(index=False)
        }
        self.data['oligo_id'] = self.data.apply(
            lambda row: [
                oligo_dict[(mutation, oligo)]
                for mutation, oligo in zip(
                    row['oligo_mut'],
                    row['oligos'],
                )
            ],
            axis=1,
        )
        self.data[['oligo_id', 'oligo_mut']] = self.data.apply(
            self._sort_oligos,
            axis=1,
        )
        self._export_df_with_lists(
            self.data[['oligo_id', 'oligo_mut']].copy(),
            os.path.join(self.fasta_dir, 'cloning_sheet.csv'),
        )
class SequenceTrimmer:
    """
    Trims adapter sequences from DNA sequences, handling both forward and reverse orientations.

    Args:
        five_prime (str): 5' adapter sequence to find and trim before
        three_prime (str): 3' adapter sequence to find and trim after
        max_error_rate (float): Maximum mismatch rate allowed when matching adapters (default: 0.1)
        min_length (int): Minimum sequence length after trimming (default: 15)

    Attributes:
        five_prime (str): Uppercase 5' adapter sequence
        three_prime (str): Uppercase 3' adapter sequence
        max_error_rate (float): Maximum allowed mismatch rate
        min_length (int): Minimum allowed sequence length
    """
    def __init__(self,
                 five_prime: str,
                 three_prime: str,
                 min_length: int,
                 max_error_rate: float = 0
                 ):
        self.five_prime = five_prime.upper()
        self.three_prime = three_prime.upper()
        self.max_error_rate = max_error_rate
        self.min_length = min_length

    def _count_mismatches(self, seq1: str, seq2: str) -> int:
        """
        Count mismatches between two sequences of equal length.

        Args:
            seq1 (str): First sequence
            seq2 (str): Second sequence

        Returns:
            int: Number of mismatched positions
        """
        return sum(c1 != c2 for c1, c2 in zip(seq1, seq2))

    def _reverse_complement(self, seq: str) -> str:
        """
        Generate reverse complement of a DNA sequence.

        Args:
            seq (str): Input DNA sequence

        Returns:
            str: Reverse complement sequence
        """
        seq = seq.upper()
        complement = {'A':'T', 'T':'A', 'G':'C', 'C':'G'}
        return ''.join(complement.get(base, base) for base in reversed(seq))

    def _find_with_mismatches(self, sequence: str, pattern: str) -> Optional[Tuple[Tuple[int, int], str]]:
        """
        Find pattern in sequence and its reverse complement, allowing mismatches.

        Args:
            sequence (str): Input sequence to search
            pattern (str): Pattern to find

        Returns:
            Optional[Tuple[Tuple[int, int], str]]: Tuple of ((start, end), strand) if found, None if not found
        """
        sequence = sequence.upper()
        pattern_len = len(pattern)

        if len(sequence) < pattern_len:
            return None

        scores = {}
        rev_comp = self._reverse_complement(sequence)

        for i, seq in enumerate([sequence, rev_comp]):
            for start in range(len(seq) - pattern_len + 1):
                window = seq[start:start + pattern_len]
                score = self._count_mismatches(window, pattern)
                scores[(start, start + pattern_len), "fwd" if i == 0 else "rev"] = score

        if not scores:
            return None

        best_pos = min(scores.items(), key=lambda x: x[1])
        return best_pos[0] if best_pos[1] <= (1 - self.max_error_rate) * pattern_len else None

    def _trim_record(self, seq: str) -> Optional[str]:
        """
        Trim adapters from a single sequence.

        Args:
            seq (str): Input DNA sequence

        Returns:
            Optional[str]: Trimmed sequence if successful, None if discarded
        """
        if len(seq) < self.min_length:
            return None

        sequence = seq
        sequence_rev_comp = self._reverse_complement(sequence)
        start = 0
        end = len(sequence)
        strand = "fwd"

        five_prime_pos = self._find_with_mismatches(sequence, self.five_prime)
        if five_prime_pos:
            start = five_prime_pos[0][0]
            strand = five_prime_pos[1]

        three_prime_pos = self._find_with_mismatches(sequence, self.three_prime)
        if three_prime_pos:
            end = three_prime_pos[0][1]

        # check if start position is less than end position
        if start < end:

            if end - start < self.min_length:
                return None

            return sequence[start:end] if strand == "fwd" else sequence_rev_comp[start:end]

        # if start is greater than end, then the region of interest is wrapping around (given the sequence is circular)
        else:
            if strand == "fwd":
                trim = sequence[start:] + sequence[:end]
            else:
                trim = sequence_rev_comp[start:] + sequence_rev_comp[:end]

            return trim

    def trim_file(self, input, input_type: str = 'fasta') -> Optional[list]:
        """
        Process FASTQ file and output trimmed sequences.

        Args:
            input: Path to input FASTQ file or FASTA file or list of either (fasta, fastq, fasta list, fastq list)
            input_type (str): Type of input, either 'fastq' or 'fasta'

        Returns:
            Optional[list]: List of trimmed sequences if output='list', None otherwise
        """
        records_stored = []
        if input_type == 'fastq':
            records_stored = [record for record in SeqIO.parse(input, "fastq")]
            seqs = [str(record.seq) for record in SeqIO.parse(input, "fastq")]
        elif input_type == 'fasta':
            records_stored = [record for record in SeqIO.parse(input, "fasta")]
            seqs = [str(record.seq) for record in SeqIO.parse(input, "fasta")]
        elif input_type == 'fasta list':
            records_stored = [record for file in input for record in SeqIO.parse(file, "fasta")]
            seqs = [str(record.seq) for file in input for record in SeqIO.parse(file, "fasta")]
        elif input_type == 'fastq list':
            records_stored = [record for file in input for record in SeqIO.parse(file, "fastq")]
            seqs = [str(record.seq) for file in input for record in SeqIO.parse(file, "fastq")]
        with ProcessPoolExecutor(max_workers=10) as executor:
            trimmed_seqs = list(executor.map(self._trim_record, seqs))
        records = []
        for seq, record in zip(trimmed_seqs, records_stored):
            if seq is not None and len(seq) >= self.min_length:
                records.append(SeqRecord(seq=Seq(seq), id=record.id,
                                      name=record.name, description=record.description))

        if input_type == 'fasta list' or input_type == 'fastq list':
            SeqIO.write(records, f"seqs_trimmed.fasta", "fasta")
        else:
            SeqIO.write(records, f"{input.split('.')[0]}_trimmed.fasta", "fasta")


class BaseProteinCDSAnalyzer:
    """
    Analyzes coding sequences (CDS) of proteins.
    Args:
        seqs (str or list): Path to FASTA file or list of sequences.
        ref_seqs (str or list): Path to reference FASTA file or list of reference sequences.
        input_type (str): Type of input, either 'fasta' or 'list'.
    """

    def __init__(self, seqs, ref_seqs, input_type='fasta'):
        self._load_sequences(seqs, ref_seqs, input_type)
        self._run_pipeline()

    def _load_sequences(self, seqs, ref_seqs, input_type):
        """
        Loads sequences from input file or list.
        Args:
            seqs (str or list): Path to FASTA file or list of sequences.
            ref_seqs (str or list): Path to reference FASTA file or list of reference sequences.
            input_type (str): Type of input, either 'fasta' or 'list'.
        """
        if input_type == 'fasta':
            self.data = pd.DataFrame([str(record.seq).upper() for record in SeqIO.parse(seqs, "fasta")], columns=['seqs'])
            self.ref_seq = str(next(SeqIO.parse(ref_seqs, "fasta")).seq).upper()
        elif input_type == 'list':
            self.data = pd.DataFrame(seqs, columns=['seqs'])
            self.ref_seq = ref_seqs[0]

    def _align_sequences(self, query_sequence):
        """
        Aligns a query sequence to the reference sequence.

        Args:
            query_sequence (str): The sequence to align.
        Returns:
            list: Aligned sequence and its length.
        """
        aligner = Align.PairwiseAligner()
        aligner.mode = 'global'
        aligner.match_score = 2
        aligner.mismatch_score = 0
        aligner.open_gap_score = -4
        aligner.extend_gap_score = -2
        alignment = next(aligner.align(self.ref_seq, query_sequence))
        return [alignment[1], len(alignment[1])]

    def _align_sequences_multithreaded(self):
        """Aligns sequences using multiple threads for improved performance."""
        with ProcessPoolExecutor() as executor:
            results = executor.map(self._align_sequences, self.data['seqs'])
        self.data[['aligned_seqs', 'aligned_seqs_length']] = pd.DataFrame(list(results))

    def _generate_mutation_name(self, input_list):
        """
        Generates a mutation name from a list of mutations.
        Args:
            input_list (list): List of mutations.

        Returns:
            str: Generated mutation name.
        """
        if not input_list:
            return 'WT'
        if input_list[0] in ['indel', 'deletion', 'contains_N']:
            return input_list[0]
        return '/'.join(sorted(input_list, key=lambda s: int(''.join(filter(str.isdigit, s)))))

    def _compare_codon_to_ref(self, sequence):
        """
        Compares codons in a sequence to the reference sequence.
        Args:
            sequence (str): The sequence to compare.
        Returns:
            tuple: Dictionary of mutation counts and dictionary of mutation details.
        """
        ref_codon_seq = [self.ref_seq[i:i+3] for i in range(0, len(self.ref_seq), 3)]
        codon_seq = [sequence[i:i+3] for i in range(0, len(sequence), 3)]

        if 'N' in sequence:
            return [0, 0, 0, 0, [], ['contains_N'], [], [], 'contains_N']

        if "-" in sequence:
            return [0, 0, 0, 0, [], ['deletion'], [], [], 'deletion']
        if len(sequence) > len(self.ref_seq):
            return [0, 0, 0, 0, [], ['indel'], [], [], 'indel']

        if len(sequence) == len(self.ref_seq):
            muts = [0, 0, 0, 0]
            seq_mutations = [[], [], [], [], '']
            for pos, (codon, ref_codon) in enumerate(zip(codon_seq, ref_codon_seq), 1):
                mismatches = sum(c1 != c2 for c1, c2 in zip(codon, ref_codon))
                if mismatches:
                    seq_mutations[mismatches].append(ref_codon + str(pos) + codon)
                muts[mismatches] += 1
            return muts + seq_mutations

    def _compare_codon_to_ref_multithreaded(self):
        """Compares codons to reference using multiple threads for improved efficiency."""
        with ProcessPoolExecutor() as executor:
            results = executor.map(self._compare_codon_to_ref, self.data['aligned_seqs'])
        self.data[['Num_Changes_0', 'Num_Changes_1', 'Num_Changes_2', 'Num_Changes_3',
                   'nt_0_mut', 'nt_1_mut', 'nt_2_mut', 'nt_3_mut', 'error']] = pd.DataFrame(list(results))

    def _convert_codon_mut_to_aa_mut(self, codon_mut_ls):
        """
        Converts codon mutations to amino acid mutations.

        Args:
            codon_mut_ls (list): List of codon mutations.
        Returns:
            list: List of amino acid mutations.
        """
        aa_mut_ls = []
        for mut in codon_mut_ls:
            if mut in ['indel', 'deletion']:
                aa_mut_ls.append(mut)
                continue
            match = re.match(r'([a-zA-Z]+)(\d+)([a-zA-Z]+)', mut)
            if match:
                part1, part2, part3 = match.groups()
                aa_i = str(Seq(part1).translate())
                aa_f = str(Seq(part3).translate())
                # Synonymous marker codons are nucleotide barcodes, not protein mutations.
                if aa_i != aa_f:
                    aa_mut_ls.append(aa_i + part2 + aa_f)
        return [aa_mut_ls]

    def _convert_codon_mut_to_aa_mut_multithreaded(self):
        """Converts codon mutations to amino acid mutations using multiple threads for better performance."""
        with ProcessPoolExecutor() as executor:
            results = executor.map(self._convert_codon_mut_to_aa_mut, self.data['codon_mut_ls'])
        self.data['aa_mut_ls'] = pd.DataFrame(list(results))
        self.data['aa_mutation'] = self.data['aa_mut_ls'].apply(self._generate_mutation_name)

    def _generate_mutation_names_all(self):
        """Generates mutation names for all sequences in the dataset."""
        self.data['codon_mut_ls'] = self.data['nt_1_mut'] + self.data['nt_2_mut'] + self.data['nt_3_mut']
        self.data['codon_mutation'] = self.data['codon_mut_ls'].apply(self._generate_mutation_name)

    def _run_pipeline(self):
        """Executes the full analysis pipeline."""
        self._align_sequences_multithreaded()
        self._compare_codon_to_ref_multithreaded()
        self._generate_mutation_names_all()
        self._convert_codon_mut_to_aa_mut_multithreaded()
        self.mutants = self.data[['aa_mut_ls','aa_mutation']]


class RawNanoporeProteinCDSAnalyzer(BaseProteinCDSAnalyzer):
    """
    Manages raw nanopore sequencing data with high error rate.
    Inherits from BaseProteinCDSAnalyzer.
    """

    def _remove_insertions(self, reference_aligned, query_aligned):
        """
        Removes insertions from aligned query sequence.

        Args:
            reference_aligned (str): Aligned reference sequence.
            query_aligned (str): Aligned query sequence.

        Returns:
            str: Query sequence with insertions removed.
        """
        return ''.join(char for i, char in enumerate(query_aligned) if reference_aligned[i] != '-')

    def _align_sequences(self, query_sequence):
        """
        Aligns a query sequence to the reference sequence, removing insertions.

        Args:
            query_sequence (str): The sequence to align.
        Returns:
            list: Aligned sequence without insertions and its length.
        """
        aligner = Align.PairwiseAligner()
        aligner.mode = 'global'
        aligner.match_score = 2
        aligner.mismatch_score = 0
        aligner.open_gap_score = aligner.extend_gap_score = -2
        alignment = next(aligner.align(self.ref_seq, query_sequence))
        query_aligned_no_ins = self._remove_insertions(*alignment)
        return [query_aligned_no_ins, len(query_aligned_no_ins)]

    # def _generate_mutation_names_all(self):
    #     """Generates mutation names for all sequences, considering only 2 and 3 nucleotide changes."""
    #     self.data['codon_mut_ls'] = self.data['nt_2_mut'] + self.data['nt_3_mut']
    #     self.data['codon_mutation'] = self.data['codon_mut_ls'].apply(self._generate_mutation_name)

    def _compare_codon_to_ref(self, sequence):
        """
        Compares codons in a sequence to the reference sequence, ignoring deletions within codons.
        Args:
            sequence (str): The sequence to compare.
        Returns:
            tuple: Dictionary of mutation counts and dictionary of mutation details.
        """
        ref_codon_seq = [self.ref_seq[i:i+3] for i in range(0, len(self.ref_seq), 3)]
        codon_seq = [sequence[i:i+3] for i in range(0, len(sequence), 3)]

        muts = [0, 0, 0, 0]
        seq_mutations = [[], [], [], [], '']
        for pos, (codon, ref_codon) in enumerate(zip(codon_seq, ref_codon_seq), 1):
            mismatches = sum(c1 != c2 for c1, c2 in zip(codon, ref_codon))
            if mismatches:
                seq_mutations[mismatches].append(ref_codon + str(pos) + codon)
            muts[mismatches] += 1
        return muts + seq_mutations
