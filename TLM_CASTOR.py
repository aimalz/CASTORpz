from pzflow import Flow
import jax.numpy as jnp
import pandas as pd 
import numpy as np

from astropy.table import Table

from pzflow import Flow, FlowEnsemble
from pzflow.distributions import Uniform
from pzflow.bijectors import Chain, StandardScaler, NeuralSplineCoupling


# load data

def getTrueY(test_cat, mag_col_names, y_col_name):

    test_cat = Table(test_cat, masked=True, copy=True)
    # remove nans
    for col in mag_col_names:
        test_cat[col].mask = np.isnan(test_cat[col].data) | test_cat[col].mask
        test_cat = test_cat[~test_cat[col].mask] # then remove nans from test set
            
    true_y = test_cat[y_col_name]
    return true_y.filled()

unp = Table.read("unperturbed_mags.fits")
mock = Table.read("ext_phot.fits")
mask = unp["Euclid_VIS_MAG"] <= 24.5

CASTOR_baseline = mock[mask] #ignoring training data outside the Euclid flux limit 

names_phot = ["ID", "photoz",
              "LSST_g_MAG", "LSST_g_MAGERR", 
              "LSST_r_MAG", "LSST_r_MAGERR",
              "LSST_i_MAG", "LSST_i_MAGERR", 
              "LSST_z_MAG", "LSST_z_MAGERR",
              "castor_uv_MAG", "castor_uv_MAGERR", 
              "castor_u_MAG", "castor_u_MAGERR", 
              "castor_g_MAG", "castor_g_MAGERR"]

available_os = ["baseline"]
names = [
    "baseline",
]

os_names = dict(zip(available_os, names))
colors = ["k"] #, "plum", "cornflowerblue", "#2ca02c", "gold", "tomato"]
os_colors = dict(zip(available_os, colors))

# put data in expected format for TLM 

LSST_g_mag = getTrueY(test_cat=CASTOR_baseline, mag_col_names=names_phot, y_col_name="LSST_g_MAG")
LSST_r_mag = getTrueY(test_cat=CASTOR_baseline, mag_col_names=names_phot, y_col_name="LSST_r_MAG")
LSST_i_mag = getTrueY(test_cat=CASTOR_baseline, mag_col_names=names_phot, y_col_name="LSST_i_MAG")
LSST_z_mag = getTrueY(test_cat=CASTOR_baseline, mag_col_names=names_phot, y_col_name="LSST_z_MAG")

LSST_g_mag_ERR = getTrueY(test_cat=CASTOR_baseline, mag_col_names=names_phot, y_col_name="LSST_g_MAGERR")
LSST_r_mag_ERR = getTrueY(test_cat=CASTOR_baseline, mag_col_names=names_phot, y_col_name="LSST_r_MAGERR")
LSST_i_mag_ERR = getTrueY(test_cat=CASTOR_baseline, mag_col_names=names_phot, y_col_name="LSST_i_MAGERR")
LSST_z_mag_ERR = getTrueY(test_cat=CASTOR_baseline, mag_col_names=names_phot, y_col_name="LSST_z_MAGERR")

CASTOR_g_mag = getTrueY(test_cat=CASTOR_baseline, mag_col_names=names_phot, y_col_name="castor_g_MAG")
CASTOR_u_mag = getTrueY(test_cat=CASTOR_baseline, mag_col_names=names_phot, y_col_name="castor_u_MAG")
CASTOR_uv_mag = getTrueY(test_cat=CASTOR_baseline, mag_col_names=names_phot, y_col_name="castor_uv_MAG")

CASTOR_g_mag_ERR = getTrueY(test_cat=CASTOR_baseline, mag_col_names=names_phot, y_col_name="castor_g_MAGERR")
CASTOR_u_mag_ERR = getTrueY(test_cat=CASTOR_baseline, mag_col_names=names_phot, y_col_name="castor_u_MAGERR")
CASTOR_uv_mag_ERR = getTrueY(test_cat=CASTOR_baseline, mag_col_names=names_phot, y_col_name="castor_uv_MAGERR")

ID = getTrueY(test_cat=CASTOR_baseline, mag_col_names=names_phot, y_col_name="ID")
z_true = getTrueY(test_cat=CASTOR_baseline, mag_col_names=names_phot, y_col_name="photoz")

#df = pd.DataFrame({'CASTOR_ID': data_array_sorted_CASTOR_wID[:,0] , 'error': data_array_sorted_CASTOR_wID[:,1], 
#                   'g-r': data_array_sorted_CASTOR_wID[:,2] - data_array_sorted_CASTOR_wID[:,3], 
#                   'r-i': data_array_sorted_CASTOR_wID[:,3] - data_array_sorted_CASTOR_wID[:,4], 
#                   'i-z': data_array_sorted_CASTOR_wID[:,4] - data_array_sorted_CASTOR_wID[:,5], 
#                   'true_z': data_array_sorted_CASTOR_wID[:,6]})

catalogs = dict()
for os in available_os:
    
#     cat = pd.DataFrame({'CASTOR_ID': ID, 'z_true': z_true, 
#                              'LSST_g_mag': LSST_g_mag, 'LSST_g_mag_ERR': LSST_g_mag_ERR, 
#                              'LSST_r_mag': LSST_r_mag, 'LSST_r_mag_ERR': LSST_r_mag_ERR, 
#                              'LSST_i_mag': LSST_i_mag, 'LSST_i_mag_ERR': LSST_i_mag_ERR, 
#                              'LSST_z_mag': LSST_z_mag, 'LSST_z_mag_ERR': LSST_z_mag_ERR, 
#                              'CASTOR_uv_mag': LSST_uv_mag, 'LSST_uv_mag_ERR': LSST_uv_mag_ERR, 
#                              'CASTOR_u_mag': LSST_u_mag, 'LSST_u_mag_ERR': LSST_u_mag_ERR, 
#               'CASTOR_g_mag': LSST_g_mag, 'LSST_g_mag_ERR': LSST_g_mag_ERR})
    
    # this will need to change to accomodate multiple catalogs, e.g. LSST only or LSST + CASTOR
    
    cat = pd.DataFrame({'CASTOR_ID': ID, 'z_true': z_true, 
                             'r': LSST_r_mag,
                             'g-r': LSST_g_mag - LSST_r_mag, 
                             'r-i': LSST_r_mag - LSST_i_mag, 
                             'i-z': LSST_i_mag - LSST_z_mag, 
                             'uv-u': CASTOR_uv_mag - CASTOR_u_mag, 
                             'u-g': CASTOR_u_mag - CASTOR_g_mag
                      })
    
    catalogs[os] = cat.dropna()

# first I create a bijector chain
# the first bijection is a standard scaler - but I'm not actually using it for standard scaling
#     I set the mean and std so that it maps the redshift range (0, 3.2) onto (-5, 5), which is 
#     the domain of the NeuralSplineCoupling
# the second bijection is a NeuralSplineCoupling. I told it to expect 6 conditions,
#     which will be the r mag and the galaxy colors
bijector = Chain(
    StandardScaler(np.atleast_1d(1.6), np.atleast_1d(0.32)),
    NeuralSplineCoupling(n_conditions=6)
)

# I set the latent distribution to a Uniform over (-5, 5)
# this range was chosen to match the NeuralSplineCoupling domain
# I chose a Uniform since all of the redshifts are drawn from a compact domain
latent = Uniform(1, 5) # did the syntax here change?

# create a dictionary that will hold all the ensembles
ensembles = dict()

# create the baseline flows
for os in available_os:

    # the data column is the one that is sampled and transformed by the flow
    data_columns = ["z_true"]
    # the conditional columns are the columns that the flow is conditioned on
    conditional_columns = ["r", "u-g", "g-r", "r-i", "i-z", "uv - u"] # different colors than LSST  

    # save some info with the flow
    info = f"Models z_true conditioned on galaxy colors and r mag from os {os}. K=16"

    # instantiate and save the flow
    flowEns = FlowEnsemble(data_columns = data_columns, 
                           conditional_columns = conditional_columns,
                           bijector = bijector,
                           latent = latent,
                           info = info,
                           N = 10)

    ensembles[os] = flowEns
    
    
    
    

for os, ens in ensembles.items():
    
    # get the data and make a train and test set
    cat = catalogs[os]
    cat_train = cat.sample(frac = 0.8)
    cat_test = cat.drop(cat_train.index)
    
    # train the flow on the given learning rate schedule
    loss1 = ens.train(cat_train, sample_errs=True,
                       optimizer = adam(step_size = 1e-3),
                       epochs = 100, seed = 123)
    loss2 = ens.train(cat_train, sample_errs=True,
                       optimizer = adam(step_size = 2e-4),
                       epochs = 100, seed = 312)
    loss3 = ens.train(cat_train, sample_errs=True,
                       optimizer = adam(step_size = 1e-4),
                       epochs = 50, seed = 231)
    
    losses = {fname : # for each flow trained in the ensemble...
                  [float(loss) # save the list of training losses
                   for lossDict in [loss1, loss2, loss3]
                   for loss in lossDict[fname]]
              for fname in loss1}
    
    # print the train and test loss
    train_loss = -np.mean(ens.log_prob(cat_train))
    test_loss = -np.mean(ens.log_prob(cat_test))
    print(os, train_loss, test_loss)
    
    # save the ensemble
    ens.save(f"trained_flows/pzflow_ensemble_for_{os}.pkl")
    # and the losses
    with open(f"trained_flows/losses_for_{os}.pkl", "wb") as file:
        pickle.dump({"losses": losses, 
                     "train loss": train_loss, 
                     "test loss": test_loss},
                    file)
