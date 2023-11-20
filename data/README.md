directory structure

```sh
data/
--experiment_id
---<cat1><cat2>_f
----<dataset_identifier>_finetune.txt
----<dataset_identifier>_different_ctg0_dev.txt
----<dataset_identifier>_identical_ctg0_dev.txt
----<dataset_identifier>_different_ctg1_dev.txt
----<dataset_identifier>_identical_ctg1_dev.txt
... and so on
```

For reasons unknown, the `identical` and `different` datasets can be set to be the same (as in copy-paste them). The results on different matter for the original work, but the code is currently hard-coded to include identical analyses (which are mysterious). So we can leave it as is.
