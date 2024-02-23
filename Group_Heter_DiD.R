library("ggplot2")
library("arrow")
library("did")
library('patchwork')

df <- read_parquet("~/Dropbox/Airbnb/proc_data/For_DiD11.parquet", header = TRUE)
# df <- df[df$QuarterID >= df$First_Treated - 10, ] #for smoothing


#df <- read_parquet("~/Dropbox/Airbnb/proc_data/For_DiD33.parquet", header = TRUE)
df <- transform(df, pp_id = as.numeric(pp_id))
df$log_Revenue = log(df$Revenue+1)
df$log_Price = log(df$Avg_Price+1)
colnames(df) <- gsub(" ", ".", colnames(df))

g_att_group <- function(out_come,df){
  example_attgt <- att_gt(yname = out_come,
                          tname = "QuarterID",
                          idname = "pp_id",
                          allow_unbalanced_panel = TRUE,
                          gname = "First_Treated",
                          xformla = ~1,
                          data = df)
  
  agg.simple <- aggte(example_attgt,na.rm = TRUE, type = "simple")
  return(agg.simple)
}

gruop_did <- function(out_come,df,group){
  if (group=='high'){
    print('======================high group')
    g = 1
  }else if(group=='mid'){
    print('======================mid group')
    g = 2
  }else if(group=='low'){
    print('======================low group')
    g = 3
  }
  g <- df[df$occu_group==g, ]
 
  agg <- g_att_group(out_come,g)
  alp <- agg$DIDparams$alp
  pointwise_cval <- qnorm(1-alp/2)
  overall_cband_upper <- agg$overall.att + pointwise_cval*agg$overall.se
  overall_cband_lower <- agg$overall.att - pointwise_cval*agg$overall.se
  return(list("att" = agg$overall.att, "se" =agg$overall.se, "ci_h" = overall_cband_upper,
              "ci_l"=overall_cband_lower))
}

all_groups <- function(out_come,df){
  h <- gruop_did(out_come,df,'high')
  m <- gruop_did(out_come,df,'mid')
  l <- gruop_did(out_come,df,'low')
  data <- data.frame(group = c("high","mid","low"),effect = c(h$att,m$att,l$att),
                     lower = c(h$ci_l,m$ci_l,l$ci_l),
                     upper = c(h$ci_h,m$ci_h,l$ci_h))
  print(data)
  p <- ggplot(data, aes(group, effect, col=group)) +        # ggplot2 plot with confidence intervals
    scale_color_manual(values=c("high" = "#00AFBB","mid"="#E7B800","low"="#FC4E07"))+
    geom_point() +
    geom_errorbar(aes(ymin = lower, ymax = upper))+ 
    scale_x_discrete(name =out_come, limits=c("high","mid","low"))+
    theme_minimal()+
    theme(legend.position="none",axis.title.y = element_blank())
  return (p)
}

p1 <- all_groups("Occupancy.Rate",df)
p2 <- all_groups("log_Revenue",df)
p3 <- all_groups("log_Price",df)

