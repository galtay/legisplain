# legisplain


Tools to bulk download and process legislative XML.


# Data Refresh Playbook


## Clone and install this repo

```bash
git clone git@github.com:galtay/legisplain.git
pip install -e legisplain
```

## Update main config

Copy the template config file

```bash
cp config_template.yml config.yml
```

Update the variables in `config.yml`
For example,

```yaml
---
bulk_path: "/home/user/data/congress-bulk"
s3_bucket: "hyperdemocracy"
```

## Write child scripts

```bash
python main.py write-scripts
```

## Run scripts

```bash
cd scripts/bulk_download
```


## run the scripts

```bash
cd scripts/bulk_download
./download-and-sync-with-log.sh
```
