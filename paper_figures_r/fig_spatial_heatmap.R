# Figura: heatmap LOSO-CV R2 por modelo x estacion (mayoria negativo = no generaliza)
setwd("/home/franciscoparrao/proyectos/Contaminacion/PM25_Santiago/paper_figures_r")
source("theme_paper.R"); setup_paper_theme("elsevier")
suppressPackageStartupMessages({library(ggplot2); library(dplyr); library(scico)})

d <- read.csv("../data/processed/spatial_models_results.csv")
# ordenar modelos y estaciones por R2 medio (peor abajo/izquierda)
mo <- d %>% group_by(model) %>% summarise(m=mean(r2)) %>% arrange(m)
so <- d %>% group_by(station) %>% summarise(m=mean(r2)) %>% arrange(m)
d$model <- factor(d$model, levels=mo$model)
d$station <- factor(d$station, levels=so$station)
d$r2c <- pmax(pmin(d$r2, 1), -3)   # clamp para el color (outliers muy negativos)

p <- ggplot(d, aes(model, station, fill=r2c)) +
  geom_tile(color="white", linewidth=0.5) +
  geom_text(aes(label=sprintf("%.2f", r2),
                color=ifelse(r2c < -1.2, "w","b")), size=2.2, family="Liberation Sans") +
  scale_color_manual(values=c(w="white", b="grey15"), guide="none") +
  scale_fill_scico(palette="vik", midpoint=0, limits=c(-3,1),
                   name=expression(R^2), direction=1) +
  scale_x_discrete(guide=guide_axis(angle=30)) +
  labs(x=NULL, y="Held-out station") +
  theme_paper() +
  theme(panel.grid=element_blank(), axis.line=element_blank(), axis.ticks=element_blank(),
        legend.key.height=unit(22,"pt"), legend.key.width=unit(8,"pt"), legend.position="right")
save_paper(p, "figs/fig_spatial_heatmap.pdf", width_cm=13.5, height_cm=8.0)
cat("OK spatial heatmap\n")
