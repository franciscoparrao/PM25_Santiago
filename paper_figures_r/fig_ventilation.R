# Figura: mecanismo de ventilacion de cuenca (Santiago)
# (a) BLH por estacion (invierno somero); (b) log VC vs PM2.5 (r=-0.68);
# (c) dVC vs dPM en dias de transicion (r=-0.51)
setwd("/home/franciscoparrao/proyectos/Contaminacion/PM25_Santiago/paper_figures_r")
source("theme_paper.R")
setup_paper_theme("elsevier")
suppressPackageStartupMessages({library(ggplot2); library(dplyr); library(patchwork); library(scico)})

v <- read.csv("../data/processed/santiago_ventilation_raw.csv")
v$date <- as.Date(v$date)
v$wspd <- sqrt(v$u^2 + v$v^2); v$VC <- v$blh*v$wspd; v$logVC <- log(pmax(v$VC,1))

pm <- read.csv("../data/processed/sinca_features_spatial.csv")
pm$date <- as.Date(pm$date)
city <- pm %>% group_by(date) %>% summarise(pm25=mean(pm25, na.rm=TRUE),
             precip=mean(era5_total_precipitation_hourly, na.rm=TRUE), .groups="drop")
m <- inner_join(city, v[,c("date","blh","VC","logVC")], by="date") %>% arrange(date)
mon <- as.integer(format(m$date,"%m"))
m$season <- factor(ifelse(mon %in% c(12,1,2),"Summer",
                   ifelse(mon %in% c(3,4,5),"Autumn",
                   ifelse(mon %in% c(6,7,8),"Winter","Spring"))),
                   levels=c("Summer","Autumn","Winter","Spring"))
m$dPM <- c(NA, diff(m$pm25)); m$dlogVC <- c(NA, diff(m$logVC))
thr <- quantile(abs(m$dPM), 0.88, na.rm=TRUE)
tr <- m[!is.na(m$dPM) & abs(m$dPM)>thr, ]

wong_season <- c(Summer="#E69F00", Autumn="#D55E00", Winter="#0072B2", Spring="#009E73")

# (a) BLH por estacion
pa <- ggplot(m, aes(season, blh, fill=season)) +
  geom_boxplot(outlier.size=0.3, outlier.alpha=0.3, linewidth=0.3, width=0.6) +
  scale_fill_manual(values=wong_season, guide="none") +
  labs(x=NULL, y="Boundary-layer height (m)") +
  theme_paper()

# (b) log VC vs PM2.5 (hexbin por densidad)
pb <- ggplot(m, aes(logVC, pm25)) +
  geom_bin2d(bins=40) +
  scale_fill_scico(palette="batlow", name="Days", direction=-1, trans="log10") +
  geom_smooth(method="lm", se=FALSE, color="#D55E00", linewidth=0.6) +
  annotate("text", x=Inf, y=Inf, hjust=1.1, vjust=1.6, size=2.6,
           family="Liberation Sans", label="r = -0.68") +
  labs(x="log ventilation coefficient", y=expression(PM[2.5]~"("*mu*g~m^{-3}*")")) +
  theme_paper() + theme(legend.key.height=unit(8,"pt"), legend.key.width=unit(10,"pt"))

# (c) dVC vs dPM en transicion
pc <- ggplot(tr, aes(dlogVC, dPM)) +
  geom_hline(yintercept=0, color="grey80", linewidth=0.3) +
  geom_vline(xintercept=0, color="grey80", linewidth=0.3) +
  geom_point(alpha=0.5, size=1.1, color="#0072B2") +
  geom_smooth(method="lm", se=FALSE, color="#D55E00", linewidth=0.6) +
  annotate("text", x=Inf, y=Inf, hjust=1.1, vjust=1.6, size=2.6,
           family="Liberation Sans", label="r = -0.51") +
  labs(x=expression(Delta*" log ventilation coeff."),
       y=expression(Delta*PM[2.5]~"("*mu*g~m^{-3}*")")) +
  theme_paper()

p <- pa | pb | pc
p <- p + plot_annotation(tag_levels="a", tag_suffix=")") &
     theme(plot.tag=element_text(face="bold", family="Liberation Sans", size=10))

save_paper(p, "figs/fig_ventilation.pdf", width_cm=18.0, height_cm=6.2)
cat(sprintf("\nBLH invierno %.0f vs verano %.0f | n=%d | transicion n=%d (thr=%.1f)\n",
    mean(m$blh[m$season=="Winter"]), mean(m$blh[m$season=="Summer"]), nrow(m), nrow(tr), thr))
