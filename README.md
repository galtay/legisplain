# legisplain


Tools to bulk download and process legislative XML.


# Data Refresh Playbook


## Clone and insteall this repo

```bash
git clone git@github.com:galtay/legisplain.git
pip install -e legisplain
```

## Update bulk govinfo download config

Navigate to the bulk download scripts,

```bash
cd scripts/bulk_download
```

Copy the template config file

```bash
cp config_template.yml config.yml
```

Update the paths in `config.yml` to match your local directory structure.
For example,

```yaml
---
output:
  data: "/home/user/data/congress-bulk/data"
  cache: "/home/user/data/congress-bulk/cache"
```

## update the scripts

```bash
python main.py write-scripts
```


## run the scripts

```bash
cd scripts/bulk_download
./download-and-sync-with-log.sh
```
