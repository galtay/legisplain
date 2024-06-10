---
configs:
  - config_name: default
    data_files:
      - split: '108'
        path: data/usc-108-billstatus.parquet
      - split: '109'
        path: data/usc-109-billstatus.parquet
      - split: '110'
        path: data/usc-110-billstatus.parquet
      - split: '111'
        path: data/usc-111-billstatus.parquet
      - split: '112'
        path: data/usc-112-billstatus.parquet
      - split: '113'
        path: data/usc-113-billstatus.parquet
      - split: '114'
        path: data/usc-114-billstatus.parquet
      - split: '115'
        path: data/usc-115-billstatus.parquet
      - split: '116'
        path: data/usc-116-billstatus.parquet
      - split: '117'
        path: data/usc-117-billstatus.parquet
      - split: '118'
        path: data/usc-118-billstatus.parquet
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
* [usc-textversion-xml](https://huggingface.co/datasets/hyperdemocracy/usc-textversion-xml) (different text versions of bills in xml)
* [usc-unified-xml](https://huggingface.co/datasets/hyperdemocracy/usc-unified-xml) (combined metadata and text version xml)

# BILLSTATUS (metadata for congresses 108-118)

* https://www.govinfo.gov/bulkdata/BILLSTATUS
* https://github.com/usgpo/bill-status/blob/main/BILLSTATUS-XML_User_User-Guide.md
* https://github.com/usgpo/bulk-data/blob/main/Bills-XML-User-Guide.md

These xml files contain metadata about each bill and
pointers to different xml files that contain various text versions of each bill.


# Column Descriptions

  | Column | Description |
  |--------|-------------|
  | legis_id | a unique ID for each bill (`{congress_num}-{legis_type}-{legis_num}`) |
  | congress_num | the congress number for the bill |
  | legis_type | one of [`hr`, `hres`, `hconres`, `hjres`, `s`, `sres`, `sconres`, `sjres`] (see [govinfo - types of legislation](https://www.govinfo.gov/help/bills)) |
  | legis_num | bills in each congress and of each type get an incrementing number as part of their ID |
  | bulk_path | XML file path during bulk download |
  | lastmod | lastmod date during bulk download |
  | bs_xml | contents of billstatus XML file |
  | bs_json| billstatus XML parsed into JSON |


# Examples

The dataset is broken into splits (one split per congress number).

```python
from datasets import load_dataset

# load each split into a `DatasetDict` keyed on congress number
dsd = load_dataset(path="hyperdemocracy/usc-billstatus")

# load a single congress number into a `Dataset`
ds = load_dataset(path="hyperdemocracy/usc-billstatus", split=117)

# load all congress numbers into a single `Dataset`
ds = load_dataset(path="hyperdemocracy/usc-billstatus", split="all")
```


# Congress Number to Date Mapping

| Congress Number | Years | Metadata | Text |
|-----------------|-------|----------|------|
| 118             | 2023-2024 | True | True |
| 117             | 2021-2022 | True | True |
| 116             | 2019-2020 | True | True |
| 115             | 2017-2018 | True | True |
| 114             | 2015-2016 | True | True |
| 113             | 2013-2014 | True | True |
| 112             | 2011-2012 | True | False |
| 111             | 2009-2010 | True | False |
| 110             | 2007-2008 | True | False |
| 109             | 2005-2006 | True | False |
| 108             | 2003-2004 | True | False |
