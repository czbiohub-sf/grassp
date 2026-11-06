# Developer instructions for documentation

## Adding functions to docs

Manually add the function to the `.md` files in `source/api` if you want it to be displayed in the docs.


## Building the documentation

```
pip install -e '.[docs]'
make docs            # from the repo root; writes docs/build/html
```

The build executes the tutorial notebooks, and a notebook that raises fails the build.
Outputs are cached in `docs/build/.jupyter_cache`, so only changed notebooks re-run.

The R tutorial (`source/tutorials/proloc_r_tutorial.Rmd`) needs R with the `ir` Jupyter kernel
(`make setup-r`). Without it, build with `GRASSP_DOCS_SKIP_R=1 make docs`: the page is then
rendered without outputs.

## Deployment

`.github/workflows/docs.yml` builds the docs on every pull request and publishes them to
<https://czbiohub-sf.github.io/grassp/> on every push to `main`. Nothing needs to be uploaded by
hand.
