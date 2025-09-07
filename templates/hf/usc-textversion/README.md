---
configs:
  - config_name: default
    data_files:
      - split: '113'
        path: data/usc-113-textversion.parquet
      - split: '114'
        path: data/usc-114-textversion.parquet
      - split: '115'
        path: data/usc-115-textversion.parquet
      - split: '116'
        path: data/usc-116-textversion.parquet
      - split: '117'
        path: data/usc-117-textversion.parquet
      - split: '118'
        path: data/usc-118-textversion.parquet
      - split: '119'
        path: data/usc-119-textversion.parquet
license: mit
language:
- en
---

# Dataset Description

This dataset is part of a family of datasets that provide convenient access to
congressional data from the US [Government Publishing Office](https://www.gpo.gov/)
via the [GovInfo Bulk Data Repository](https://www.govinfo.gov/developers).
GovInfo provides bulk data in xml format. 
The raw xml files were downloaded using the 
[congress](https://github.com/unitedstates/congress) repo.
Further processing was done using the
legisplain [legisplain](https://github.com/galtay/legisplain) repo.

# Hyperdemocracy Datasets

* [usc-billstatus](https://huggingface.co/datasets/hyperdemocracy/usc-billstatus) (metadata on each bill)
* [usc-textversion](https://huggingface.co/datasets/hyperdemocracy/usc-textversion) (different text versions of bills in xml)
* [usc-unified](https://huggingface.co/datasets/hyperdemocracy/usc-unified) (combined metadata and text version xml)


# TEXTVERSIONS (text for congresses 113-119)

* https://www.govinfo.gov/bulkdata/BILLS
* https://xml.house.gov/
* https://github.com/usgpo/bill-dtd?tab=readme-ov-file

These xml files contain multiple text versions for each bill.


# Column Descriptions

  | Column | Description |
  |--------|-------------|
  | tv_id | a unique ID for each text version (`{congress_num}-{legis_type}-{legis_num}-{legis_version}-{xml_type}`) |
  | legis_id | a unique ID for each bill (`{congress_num}-{legis_type}-{legis_num}`) |
  | congress_num | the congress number for the bill |
  | legis_type | one of [`hr`, `hres`, `hconres`, `hjres`, `s`, `sres`, `sconres`, `sjres`] (see [govinfo - types of legislation](https://www.govinfo.gov/help/bills)) |
  | legis_num | bills in each congress and of each type get an incrementing number as part of their ID |
  | legis_version | version of bill text (see [govinfo - common versions of bills](https://www.govinfo.gov/help/bills)) |
  | legis_class | one of [`bills`, `plaw`] |
  | bulk_path | XML file path during bulk download |
  | file_name | last part of bulk_path. used in joining to billstatus |
  | lastmod | lastmod date during bulk download |
  | xml_type | one of [`dtd`, `uslm`] |
  | root_tag | the root xml tag. one of [`bill`, `resolution`, `amendment-doc`, `pLaw`]|
  | tv_xml | contents of textversion XML file |
  | tv_txt | a plain text version of the XML content |



# Examples

The dataset is broken into splits (one split per congress number).

```python
from datasets import load_dataset

# load each split into a `DatasetDict` keyed on congress number
dsd = load_dataset(path="hyperdemocracy/usc-textversion")

# load a single congress number into a `Dataset`
ds = load_dataset(path="hyperdemocracy/usc-textversion", split=117)

# load all congress numbers into a single `Dataset`
ds = load_dataset(path="hyperdemocracy/usc-textversion", split="all")
```


# Congress Number to Date Mapping

| Congress Number | Years | Metadata | Text |
|-----------------|-------|----------|------|
| 119             | 2025-2026 | True | True |
| 118             | 2023-2024 | True | True |
| 117             | 2021-2022 | True | True |
| 116             | 2019-2020 | True | True |
| 115             | 2017-2018 | True | True |
| 114             | 2015-2016 | True | True |
| 113             | 2013-2014 | True | True |
