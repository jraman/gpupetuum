#
library(data.table)
library(lattice)

args = commandArgs(TRUE)
filename = args[1]
# filename = 'timit_c2001_l6_2048.1.0.log'
pngfile = sub("[.]log", ".pdf", filename)
if (filename == pngfile) stop("filename and pngfile are the same!")
print(sprintf("Saving to file %s", pngfile))
lr = sub("[.]log", "", sub("[^.]+.", "", basename(filename)))
ylim = c(3, 8)
main_title = sprintf("timit, eta=%s, epochs=40", lr)
df = read.table(filename, sep='\a')
vec1 = as.vector(df[grep(' minibatch_avg_cost=', df$V1), ])
dt = data.table(matrix(unlist(strsplit(vec1, "[ \t,]+")), nrow=length(vec1), byrow=T))
setnames(dt, c("V7", "V14"), c("time", "cost"))
dt$time = as.numeric(sub("delta_t=", "", dt$time))
dt$cost = as.numeric(sub("minibatch_avg_cost=", "", dt$cost))
pdf(pngfile)
xyplot(cost ~ time, dt, xlab="time (sec)", type=c("spline"), lwd=3, grid=T, main = main_title,
       ylim = ylim)
dev.off()
