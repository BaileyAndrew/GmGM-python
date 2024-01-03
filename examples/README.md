# Examples

## Single-Cell Transcriptomics: Matrix-Variate Unimodal

Modalities:
- scRNA (Cells x Genes)

Axes:
- Cells
- Genes

For an example of how to use our algorithm on this data, see `example.ipynb` (TODO: update example).

## Duck Video: Tensor-Variate Unimodal

Modalities:
- Video (Frames x Rows x Columns)

Axes:
- Frames
- Rows
- Columns

## 10x Multiome scRNA+scATAC: Matrix-Variate Multimodal

Modalities:
- scRNA (Cells x Genes)
- scATAC (Cells x Peaks)

Axes:
- Cells
- Genes
- Peaks

For an example of how to use our algorithm on this data, see TODO

## Multi-Patient scRNA: Matrix-Variate Multimodal

Modalities:
- scRNA for Patient 1 (Cells 1 x Genes)
- scRNA for Patient 2 (Cells 2 x Genes)
- ...
- scRNA for Patient N (Cells N x Genes)

Axes:
- Genes
- Cells 1
- Cells 2
- ...
- Cells N

### Addendum

Sometimes, the data can be more complex then this structure; suppose our patients were from 2 cohorts: a control group and a group treated by some drug.  We would likely expect the drug to have some effect on gene expression, and hence it makes sense to consider two Genes axes, one for each cohort.  These cohort-specific axes would have some constraint requiring them to look similar to eachother.  **This scenario is currently out-of-scope for GmGM**.

## Multi-Patient Bluk Omics: Matrix-Variate Multimodal

Modalities:
- Metagenomics (Patient x Species)
- Metabolomics (Patient x Metabolites)

Axes:
- Patient
- Species
- Metabolites

For an example of how to use our algorithm on this data, see TODO

### Addendum

It is not uncommon for a subset of patients to only be in one modality.  In this case, axes may have "partial overlap".  **This scenario is currently out-of-scope for GmGM**.

## Synthetic Data

You can also use this package to generate synthetic data.  For an example, see `synthetic_data.ipynb`.  TODO: Add DatasetGenerator.