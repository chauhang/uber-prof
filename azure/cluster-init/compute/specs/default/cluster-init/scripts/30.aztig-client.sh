#!/bin/bash

#Read aztig configuration file including (at least) INFLUXDB_USER, INFLUXDB_PWD and GRAFANA_SHARED
source "$CYCLECLOUD_SPEC_PATH"/files/config/aztig.conf

if [ -z "$GRAFANA_SHARED" ]; then
    echo "Grafana shared folder parameter is required"
    exit 1
fi
if [ -z "$INFLUXDB_USER" ]; then
    echo "InfluxDB user parameter is required"
    exit 1
fi
if [ -z "$INFLUXDB_PWD" ]; then
    echo "InfluxDB password parameter is required"
    exit 1
fi

GRAFANA_SERVER=$(cat "$GRAFANA_SHARED"/grafana/grafana_server.conf)
if [ -z "$GRAFANA_SERVER" ]; then
    echo "Grafana server information could not be found. Make sure the ${GRAFANA_SHARED}/grafana/grafana_server.conf is accessible."
    exit 1
fi

os=$(awk -F= '/^NAME/{print $2}' /etc/os-release)

if [[ $os = *CentOS* ]]
then 
  echo "You are running on CentOS"
  echo "#### Telegraf Installation:"
  wget https://dl.influxdata.com/telegraf/releases/telegraf-1.19.1-1.x86_64.rpm
  yum localinstall -y telegraf-1.19.1-1.x86_64.rpm

elif [[ $os = *Ubuntu* ]]
then
  echo "You are running on Ubuntu"
  echo "### Telegraf Install:"
  wget https://dl.influxdata.com/telegraf/releases/telegraf_1.19.1-1_amd64.deb
  dpkg -i telegraf_1.19.1-1_amd64.deb
else
  echo "You are running on non-support OS" 
  exit 1
fi  

echo "Push right config .... "
# Update telegraph.conf
cp /etc/telegraf/telegraf.conf /etc/telegraf/telegraf.conf.origin
cp "$CYCLECLOUD_SPEC_PATH"/files/config/telegraf.conf /etc/telegraf/

cat << EOF >> /etc/telegraf/telegraf.conf

[[outputs.influxdb]]
  urls = ["http://$GRAFANA_SERVER:8086"]
  database = "monitor"
  username = "$INFLUXDB_USER"
  password = "$INFLUXDB_PWD"
EOF

echo "#### Starting Telegraf services:"
service telegraf stop
service telegraf start
service telegraf status

echo "### Finished Telegraf setup"