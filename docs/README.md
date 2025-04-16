# Developer instructions for documentation

## Adding functions to docs

Manually add the function to the `.md` files in `source/api` if you want it to be displayed in the docs.


## Updating the documentation

```
cd docs
# make clean (optional, removes build dir)
make html
```

## Deployment

At the moment the documentation is just hosted internally on bruno under https://onsite.czbiohub.org/compbio/grassp/api/plotting.html. To upload run
```
sh upload_local.sh
```

