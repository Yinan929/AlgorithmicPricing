library("Synth")
library("arrow")
library("dplyr")
data("basque")

dataprep.out <- dataprep(foo = basque,predictors = c("school.illit",
          "school.prim", "school.med","school.high", "school.post.high", "invest"),
          predictors.op = "mean",
          time.predictors.prior = 1964:1969,
          special.predictors = list(
          list("gdpcap", 1960:1969 , "mean"),
          list("sec.agriculture", seq(1961, 1969, 2), "mean"),
          list("sec.energy", seq(1961, 1969, 2), "mean"),
          list("sec.industry", seq(1961, 1969, 2), "mean"),
          list("sec.construction", seq(1961, 1969, 2), "mean"),
          list("sec.services.venta", seq(1961, 1969, 2), "mean"),
          list("sec.services.nonventa", seq(1961, 1969, 2), "mean"),
          list("popdens", 1969, "mean")),
          dependent = "gdpcap",
          unit.variable = "regionno",
          unit.names.variable = "regionname",
          time.variable = "year",
          treatment.identifier = 17,
          controls.identifier = c(2:16, 18),
          time.optimize.ssr = 1960:1969,
          time.plot = 1955:1997)

synth.out <- synth(data.prep.obj = dataprep.out, method = "L-BFGS-B")

gaps <- dataprep.out$Y1plot - (dataprep.out$Y0plot %*% synth.out$solution.w)
synth.tables <- synth.tab(synth.res = synth.out,dataprep.res=dataprep.out)
path.plot(synth.res = synth.out, dataprep.res = dataprep.out,
        Ylab = "real per-capita GDP (1986 USD, thousand)", Xlab = "year",
        Ylim = c(0, 12), Legend = c("Basque country", "synthetic Basque country"),Legend.position="bottomright")


gaps.plot(synth.res = synth.out, dataprep.res = dataprep.out,
        Ylab = "gap in real per-capita GDP (1986 USD, thousand)", Xlab = "year",
        Ylim = c(-1.5, 1.5), Main = NA)


#c("Price.Changes",
  #"Occupancy.Rate", "Avg_Price","Number.of.Reviews", "Overall.Rating", "R")



df <- read_parquet("~/Dropbox/Airbnb/proc_data/For_Synth.parquet", header = TRUE)
colnames(df) <- gsub(" ", ".", colnames(df))
df <- transform(df, PropertyNo = as.numeric(PropertyNo))


outcome_var <-"Price.Changes"

dataprep.out <- dataprep(foo = df,predictors = c("Price.Changes", "Revenue", 
                          "Avg_Price","Occupancy.Rate","Number.of.Reviews", "Overall.Rating", "R","A"),
                           predictors.op = "mean",
                           time.predictors.prior = c(5:17),
                           dependent = outcome_var,
                           unit.variable = "PropertyNo",
                           unit.names.variable = "Property.ID",
                           time.variable = "QuarterID",
                           treatment.identifier = 10,
                           controls.identifier = c(1:9),
                           time.optimize.ssr = c(5:17),
                           time.plot = c(5:20))
synth.out <- synth(data.prep.obj = dataprep.out, method = "BFGS")

gaps <- 100*(dataprep.out$Y1plot - (dataprep.out$Y0plot %*% synth.out$solution.w))/(dataprep.out$Y0plot %*% synth.out$solution.w)
synth.tables <- synth.tab(synth.res = synth.out,dataprep.res=dataprep.out)
path.plot(synth.res = synth.out, dataprep.res = dataprep.out,
          Ylab = outcome_var, Xlab = "Quarter",Legend = c("Algorithmic", "Synthetic Property"),Legend.position="topright")
abline(v=18, col="blue")
gaps.plot(synth.res = synth.out, dataprep.res = dataprep.out,
          Ylab = outcome_var, Xlab = "Quarter", Main = NA) 
abline(v=18, col="blue")

#######################################################################
#Demean data sets
demean_data <- function(data){
  maindata <- data %>%
    group_by(MonthID) %>%
    mutate(avg_occ = mean(Occupancy.Rate, na.rm = TRUE)) %>%
    mutate(avg_price = mean(Avg_Price, na.rm = TRUE)) %>%
    mutate(avg_rev = mean(Revenue, na.rm = TRUE)) %>%
    mutate(Occupancy.Rate = Occupancy.Rate - avg_occ) %>%
    mutate(Avg_Price = Avg_Price - avg_price) %>%
    mutate(Revenue = Revenue - avg_rev)
  
  return(maindata)
}

Treated <- read_parquet("~/Dropbox/Airbnb/proc_data/Treated1.parquet", header = TRUE)
colnames(Treated) <- gsub(" ", ".", colnames(Treated))
Treated_pps <- unique(Treated$Property.ID)
Treated <-demean_data(Treated) 
Donor <- read_parquet("~/Dropbox/Airbnb/proc_data/Donor.parquet", header = TRUE)
colnames(Donor) <- gsub(" ", ".", colnames(Donor))
Donor <-demean_data(Donor) 

#Combine two data sets and find counterpart in Nonusers
find_donor_pool <- function(pp_id,Donor_df,Treated_df) {
  treated_snap <- Treated_df[Treated_df$Property.ID == pp_id,]
  period_months <- unique(treated_snap$MonthID)
  prior_time <- period_months[1:treated_snap$Consec_period_count[1]]
  number_periods <- length(period_months)
  donor <- Donor_df %>%
    filter(MonthID %in% period_months) %>%
    group_by(Property.ID) %>% 
    filter(n() ==number_periods)

  df<- rbind(donor,subset(treated_snap, select = -c(Consec_period_count,order) ))
  df$AvailableDays <- df$R + df$A
  df <- transform(df,PropertyNo = as.numeric(factor(Property.ID)))
  ids <- unique(df$PropertyNo)
  control_ids <- ids[1:length(ids)-1]
  treatment_id <- ids[length(ids)]
  return(list("data" = df, "all_time" =period_months, "prior_time" = prior_time,
              "controls"=control_ids,"treatment"=treatment_id))
}	

Synth_Control <- function(outcome_var,data,time_periods,prior_periods,controls,treatment){
  control_vars <- c("Price.Changes", "Revenue", "Bedrooms",
                   "Avg_Price","Occupancy.Rate","Number.of.Reviews", "Overall.Rating")
  control_vars <- control_vars[control_vars != outcome_var]
  dataprep.out <- dataprep(foo = data,predictors = control_vars,
                           predictors.op = "mean",
                           time.predictors.prior =prior_periods,
                          special.predictors = list(list("AvailableDays", time_periods , "mean")),
                           dependent = outcome_var,
                           unit.variable = "PropertyNo",
                           unit.names.variable = "Property.ID",
                           time.variable = "MonthID",
                           treatment.identifier = treatment,
                           controls.identifier = controls,
                           time.optimize.ssr = prior_periods,
                           time.plot = time_periods)
  synth.out <- synth(data.prep.obj = dataprep.out, method = "BFGS")
  
  synth.tables <- synth.tab(synth.res = synth.out,dataprep.res=dataprep.out)
  tb <- synth.tables$tab.w
  
  gaps <- 100*(dataprep.out$Y1plot - (dataprep.out$Y0plot %*% synth.out$solution.w))/(dataprep.out$Y0plot %*% synth.out$solution.w)
  return(list("rslt"=gaps[(length(prior_periods)+1):(length(prior_periods)+5)],"nonzero"=sum(tb$w.weights != 0))) #only keep the first 5 post treatment period
}


Multiple_Round <- function(outcome_var,list_pps,Donor_df,Treated_df){
  FNL <- data.frame(Property.ID=character(),M1=double(),M2=double(),
                    M3=double(),M4=double(),M5=double())
  for (x in list_pps) {
    print(x)
    skip_to_next <- FALSE
    donor_unit <- find_donor_pool(x,Donor_df,Treated_df)
    
    if (length(donor_unit$controls) < 10) {
      next
    } 
    
    tryCatch(effect <- Synth_Control(outcome_var,donor_unit$data,donor_unit$all_time,
                                     donor_unit$prior_time,donor_unit$controls,donor_unit$treatment)
             , error = function(e) { skip_to_next <<- TRUE})
    if(skip_to_next) { next }
    

    t <- data.frame(x,matrix(unlist(effect$rslt), ncol = length(effect$rslt)),length(donor_unit$controls),effect$nonzero)
    names(t)<-c("Property.ID","M1","M2","M3","M4","M5","Donors","Nonzero")
    FNL<-rbind(FNL,t)
    
    
  }
  return(FNL)
}


data_Revenue <- Multiple_Round('Revenue',Treated_pps,Donor,Treated)
data_Revenue_m <- data_Revenue[data_Revenue[, "Nonzero"]>=10, ]
write.csv(data_Revenue_m,"~/Dropbox/Airbnb/proc_data/Synth_Revenue.csv", row.names = FALSE)

data_Occupancy <- Multiple_Round('Occupancy.Rate',Treated_pps,Donor,Treated)
data_Occupancy_m <- data_Occupancy[data_Occupancy[, "Nonzero"]>=10, ]
write.csv(data_Occupancy_m,"~/Dropbox/Airbnb/proc_data/Synth_Occupancy.csv", row.names = FALSE)

data_Price <- Multiple_Round('Avg_Price',Treated_pps,Donor,Treated)
data_Price_m <- data_Price[data_Price[, "Nonzero"]>=10, ]
write.csv(data_Price_m,"~/Dropbox/Airbnb/proc_data/Synth_Price.csv", row.names = FALSE)

data_Changes <- Multiple_Round('Price.Changes',Treated_pps,Donor,Treated)
data_Changes_m <- data_Changes[data_Changes[, "Nonzero"]>=10, ]
write.csv(data_Changes_m,"~/Dropbox/Airbnb/proc_data/Synth_Change.csv", row.names = FALSE)






test<- find_donor_pool('ab-21242445',Donor,Treated)   
Synth_Control11 <- function(outcome_var,data,time_periods,prior_periods,controls,treatment){
  control_vars <- c("Price.Changes", "Revenue", "Bedrooms",
                    "Avg_Price","Occupancy.Rate","Number.of.Reviews", "Overall.Rating")
  control_vars <- control_vars[control_vars != outcome_var]
  dataprep.out <- dataprep(foo = data,predictors = control_vars,
                           predictors.op = "mean",
                           time.predictors.prior =prior_periods,
                           special.predictors = list(list("AvailableDays", time_periods , "mean")),
                           dependent = outcome_var,
                           unit.variable = "PropertyNo",
                           unit.names.variable = "Property.ID",
                           time.variable = "MonthID",
                           treatment.identifier = treatment,
                           controls.identifier = controls,
                           time.optimize.ssr = prior_periods,
                           time.plot = time_periods)
  synth.out <- synth(data.prep.obj = dataprep.out, method = "BFGS")
  path.plot(synth.res = synth.out, dataprep.res = dataprep.out,
            Ylab = outcome_var, Xlab = "Month",Legend = c("Algorithmic", "Synthetic Property"),Legend.position="topright")
  #abline(v=18, col="blue")
  gaps <- 100*(dataprep.out$Y1plot - (dataprep.out$Y0plot %*% synth.out$solution.w))/(dataprep.out$Y0plot %*% synth.out$solution.w)
  gaps.plot(synth.res = synth.out, dataprep.res = dataprep.out) 
  return(synth.out)
}
rslt <- Synth_Control11("Occupancy.Rate",test$data,test$all_time,test$prior_time,test$controls,test$treatment)
