# Figura: analisis estacional (walk-forward) — distribucion PM2.5 + R2 por estacion
setwd("/home/franciscoparrao/proyectos/Contaminacion/PM25_Santiago/paper_figures_r")
source("theme_paper.R"); setup_paper_theme("elsevier")
suppressPackageStartupMessages({library(ggplot2); library(dplyr); library(patchwork)})

lev <- c("Summer","Autumn","Winter","Spring")
wong_season <- c(Summer="#E69F00", Autumn="#D55E00", Winter="#0072B2", Spring="#009E73")

# (a) distribucion real de PM2.5 por estacion
pm <- read.csv("../data/processed/sinca_features_spatial.csv"); pm$date <- as.Date(pm$date)
mon <- as.integer(format(pm$date,"%m"))
pm$season <- factor(ifelse(mon%in%c(12,1,2),"Summer",ifelse(mon%in%c(3,4,5),"Autumn",
                    ifelse(mon%in%c(6,7,8),"Winter","Spring"))), levels=lev)
pa <- ggplot(pm, aes(season, pm25, fill=season)) +
  geom_boxplot(outlier.size=0.2, outlier.alpha=0.2, linewidth=0.3, width=0.6) +
  scale_fill_manual(values=wong_season, guide="none") +
  scale_y_continuous(limits=c(0,150), expand=expansion(mult=c(0,0.03))) +
  labs(x=NULL, y=expression(PM[2.5]~"("*mu*g~m^{-3}*")")) + theme_paper()

# (b) R2 walk-forward por estacion
s <- read.csv("../data/processed/apr_2_3_seasonal_walkforward.csv")
s$season <- factor(s$season, levels=lev)
pb <- ggplot(s, aes(season, r2, fill=season)) +
  geom_col(width=0.62, color="grey25", linewidth=0.2) +
  geom_hline(yintercept=0.764, linetype="dashed", color="grey35", linewidth=0.4) +
  annotate("text", x=0.6, y=0.79, label="overall 0.76", hjust=0, size=2.4,
           color="grey35", family="Liberation Sans") +
  geom_text(aes(label=sprintf("%.2f", r2)), vjust=-0.5, size=2.5, family="Liberation Sans") +
  scale_fill_manual(values=wong_season, guide="none") +
  scale_y_continuous(limits=c(0,0.95), breaks=seq(0,0.8,0.2), expand=expansion(mult=c(0,0.02))) +
  labs(x=NULL, y=expression(R^2~"(1-day, walk-forward)")) + theme_paper()

p <- (pa | pb) + plot_annotation(tag_levels="a", tag_suffix=")") &
     theme(plot.tag=element_text(face="bold", family="Liberation Sans", size=10))
save_paper(p, "figs/fig_seasonal.pdf", width_cm=15.0, height_cm=6.0)
cat("OK seasonal\n")
