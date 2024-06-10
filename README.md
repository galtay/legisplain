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
pg_conn_str: "postgresql+psycopg2://[user[:password]@][netloc][:port][/dbname][?param1=value1&...]
```

## Write child scripts

```bash
python main.py write-scripts
```

## Run scripts

```bash
cd scripts/bulk_download
./download-and-sync-with-log.sh
```

## Populate postgres

```bash
python main.py pg-populate-billstatus
python main.py pg-populate-textversion-xml
python main.py pg-populate-unified-xml
```
