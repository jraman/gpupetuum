# Parse log file of dbn_rambatch
# Sample logfile:
# 2015-09-05 03:41:16,794 root dbn_batch.py:512 INFO delta_t=3.279, epoch 1/40, mega 1/1, mini 1/50, minibatch_avg_cost=7.60140371323
# 2015-09-05 03:41:20,083 root dbn_batch.py:512 INFO delta_t=6.568, epoch 1/40, mega 1/1, mini 2/50, minibatch_avg_cost=7.59753847122
# ...
# 2015-09-05 03:43:58,138 root dbn_batch.py:512 INFO delta_t=164.623, epoch 1/40, mega 1/1, mini 50/50, minibatch_avg_cost=7.42044782639

library(data.table)
library(lattice)

args = commandArgs(TRUE)
filename = args[1]
# filename = 'timit_c2001_l6_2048.1.0.log'
pngfile = sub("[.]log", ".pdf", filename)
if (filename == pngfile) stop("filename and pngfile are the same!")
print(sprintf("Saving to file %s", pngfile))
lr = sub("[.]log", "", sub("[^.]+.", "", basename(filename)))
main_title = sprintf("timit, eta=%s, epochs=100", lr)
df = read.table(filename, sep='\a')
vec1 = as.vector(df[grep(' minibatch_avg_cost=', df$V1), ])
dt = data.table(matrix(unlist(strsplit(vec1, "[ \t,]+")), nrow=length(vec1), byrow=T))
setnames(dt, c("V7", "V14"), c("time", "cost"))
dt$time = as.numeric(sub("delta_t=", "", dt$time))
dt$cost = as.numeric(sub("minibatch_avg_cost=", "", dt$cost))
pdf(pngfile)
xyplot(cost ~ time, dt, xlab="time (sec)", type=c("spline"), lwd=3, grid=T, main = main_title,
       ylim = c(2, 8), scales = list(y=list(at=seq(1, 8))))
dev.off()
