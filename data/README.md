# Data

The data is not included in this repository. It was given to course participants of GCI World 2026
(Matsuo-Iwasawa Lab, University of Tokyo) for the final assignment and comes from an anonymous telecom
operator called "Company A".

To run the notebooks and scripts, put the two files here:

```
data/
└── telecom/
    ├── Client.csv    # 100,000 rows x 50 columns
    └── Record.csv    # 100,000 rows x 51 columns
```

## What is in the files

Both files have one row per customer and share the key `Customer_ID`.

- `Client.csv`: account and customer info, for example credit class (`crclscod`),
  handset price and age (`hnd_price`, `eqpdays`), lifetime and 3/6-month averages of usage and revenue
  (`avg3mou`, `avg6mou`, `avg3rev`, ...), and household data from outside sources (`income`, `numbcars`,
  `dwllsize`, ...).
- `Record.csv`: monthly averages of usage and billing (`rev_Mean`, `mou_Mean`, `totmrc_Mean`, `ovrrev_Mean`, ...),
  call quality (`drop_vce_Mean`, `blck_vce_Mean`, ...), customer care calls (`custcare_Mean`), tenure in
  months (`months`) and the target `churn` (1 = customer churned 31-60 days after the observation date).

About 49.6% of customers churned, so the classes are balanced. Some column meanings were not fully explained
by the data provider (for example `recv_sms_Mean` and `iwylis_vce_Mean`).
