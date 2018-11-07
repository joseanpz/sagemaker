#!/usr/bin/env Rscript
library(BVSNLP)
# library(aws.s3)

# prefix <- "s3://boi-banregio/datalake/data/InteligenciaRiesgos/M&M"

# sample_key <- "MCV/DATASETS_3/JAT_MCV_VAR_VARIABLES_SELECCION/SAMPLES/STRATEGY2_PRP.csv"

# object <- paste(prefix, sample_key, sep = "/")

# lectura <- aws.s3::s3read_using(read.csv, object=object, sep = ',')  # , header = FALSE, col.names =c()) 


# data shape: n x 3904
data <- read.csv('/tmp/strategy2_prp.csv')


y <- data[, "BMI"]

# X <- scale(data.matrix(data[, 3:3904]))
X <- data.matrix(data[, 3:3902])

print("------- Scaling -------")
X.scaled = scale(X)


print("------- Selecting features ------- ")
feat_sel <- bvs(X.scaled, y, prep = FALSE, fixed_cols = NULL, eff_size = 0.5,
                family = "logistic", hselect = FALSE, nlptype = "piMOM",
                r = 1, tau = 0.25, niter = 30, mod_prior = "beta",
                inseed = NULL, cplng = FALSE, ncpu = 4, parallel.MPI = FALSE)

coefs = unname(feat_sel$beta_hat)

feats = feat_sel$gene_names[feat_sel$HPM]

print(feats)

feat_df = data.frame(c('INTERCEPT', feats), c(coefs))
colnames(feat_df) <- c('Feature', 'Coeficient')

#zz <- rawConnection(raw(0), "r+")
write.csv(feat_df, '/tmp/bvsnlp_feat_sel.csv', row.names = FALSE)


#bvslnp_obj <- "datalake/data/InteligenciaRiesgos/M&M/MCV/DATASETS_3/JAT_MCV_VAR_VARIABLES_SELECCION/SAMPLES/FEATURE_SELECTION/BVSNLP.csv"

# upload the object to S3
#aws.s3::put_object(file = rawConnectionValue(zz),
#                   bucket = "boi-banregio", object = bvslnp_obj)

# close the connection
#close(zz)