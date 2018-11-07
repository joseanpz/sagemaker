#!/usr/bin/env Rscript
library(BVSNLP)
# library(aws.s3)

# prefix <- "s3://boi-banregio/datalake/data/InteligenciaRiesgos/M&M"

# sample_key <- "MCV/DATASETS_3/JAT_MCV_VAR_VARIABLES_SELECCION/SAMPLES/STRATEGY2_PRP.csv"

# object <- paste(prefix, sample_key, sep = "/")

# lectura <- aws.s3::s3read_using(read.csv, object=object, sep = ',')  # , header = FALSE, col.names =c()) 

data <- read.csv('/tmp/strategy2_prp.csv')

y <- data[, "BMI"]

# X <- data.matrix(lectura[, 3:3904])
X <- data.matrix(data[, 3:150])

family = "logistic"
eff_size = 0.5
nlptype = "piMOM"
mod_prior = "beta"

hyper <- HyperSelect(X, y, eff_size, nlptype, 20000, mod_prior, family)

print(hyper)