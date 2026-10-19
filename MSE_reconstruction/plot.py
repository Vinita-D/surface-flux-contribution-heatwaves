import matplotlib.pyplot as plt
Cp = 1004.0
Lv = 2.2e6
g = 9.80665


df = pd.read_csv("/jetfs/scratch/vdeshmukh/6hr_back/output_6hr/merged_hourly.csv") 



fig, ax = plt.subplots(figsize=(7,4),dpi = 200)
l1, = ax.plot(df['release_time'],df['rec_theta_e'],label='rec_theta_e',color='green')
l3, = ax.plot(df['release_time'] ,df["IC_theta_e"] + (df["h_flux"]/Cp),label='IC_theta_e + h_flux/Cp ',color='maroon' )
l4, = ax.plot(df['release_time'] ,df["IC_theta_e"],  "--",alpha = 0.4,label='IC_theta_e',color='purple')
ax.set_ylabel("K")
ax.set_xlabel("Release time")
ax.grid()
ax.tick_params(axis="x", rotation=45)

ax2 = ax.twinx()
l2, = ax2.plot(df['release_time'] ,(df["h_flux"]/Cp ), "--",alpha = 0.4,label='h_flux/Cp',color='red')
ax2.set_ylabel("flux contribution (K)")

lines = [l1, l2, l3, l4]
labels = [ln.get_label() for ln in lines]
ax.legend(lines, labels, loc="lower center", bbox_to_anchor=(0.5, 1.15),
          ncol=2, frameon=False)
plt.title("h_flux = (sen*SSHF/100) + Lv*(sen*MF/100)",fontsize=10)
fig.tight_layout(rect=[0, 0, 0.78, 1])
plt.show()

import matplotlib.pyplot as plt


fig, ax = plt.subplots(figsize=(7,4),dpi = 200)
l1, = ax.plot(df['release_time'],df['rec_mse']*1e-3,label='rec_MSE',color='green')
#l3, = ax.plot(x['release_time'] ,x['IC_mse']*1e-3 + ( Cp* x['T_flux']*1e-3) + ( Lv* x['q_flux']*1e-3) ,color='maroon', label = 'IC_MSE + Cp*T_flux + Lv*q_flux ')
l3, = ax.plot(df['release_time'] ,df["IC_mse"]*1e-3 + df['SHF_flux'] + df['MF_flux']*1e3 ,label='IC_MSE + flux_c ',color='maroon' )
l4, = ax.plot(df['release_time'] ,df['IC_mse']*1e-3,  "--",alpha = 0.4,label='IC_MSE',color='purple')
ax.set_ylabel("K")
ax.set_xlabel("Release time")
ax.grid()
ax.tick_params(axis="x", rotation=45)

ax2 = ax.twinx()
l2, = ax2.plot(df['release_time'] ,(df['SHF_flux'] + df['MF_flux']*1e3 )  , "--",alpha = 0.4,label='flux_c',color='red')
ax2.set_ylabel("flux contribution (K)")

lines = [l1, l2, l3, l4]
labels = [ln.get_label() for ln in lines]
ax.legend(lines, labels, loc="lower center", bbox_to_anchor=(0.5, 1.15),
          ncol=2, frameon=False)
plt.title("MSE = CpT + Lvq+ gz , flux_c = (sen*SSHF/Cp*100) + (sen*MF/100)",fontsize=10)
fig.tight_layout(rect=[0, 0, 0.78, 1])
plt.show()


import matplotlib.pyplot as plt


fig, ax = plt.subplots(figsize=(7,4),dpi = 200)
l1, = ax.plot(df['release_time'],df['rec_mse']*1e-3,label='rec_MSE',color='green')
l2, = ax.plot(df['release_time'],df['IC_mse']*1e-3,label='IC_MSE',color='magenta')
l3, = ax.plot(df['release_time'],df['IC_theta_e'],label='IC_theta_e',color='purple')
l4, = ax.plot(df['release_time'],df['rec_theta_e'],label='rec_theta_e',color='lightgreen')

ax.set_ylabel("K")
ax.set_xlabel("Release time")
ax.grid()
ax.tick_params(axis="x", rotation=45)
ax2 = ax.twinx()
l5, = ax2.plot(df['release_time'],df['IC_z'],label='IC_z',color='lightblue')
ax2.set_ylabel("height(m)")


lines = [l1, l2, l3, l4,l5]
labels = [ln.get_label() for ln in lines]
ax.legend(lines, labels, loc="lower center", bbox_to_anchor=(0.5, 1.15),
          ncol=2, frameon=False)
fig.tight_layout(rect=[0, 0, 0.78, 1])
plt.show()
