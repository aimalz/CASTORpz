from astropy.table import Table, MaskedColumn, vstack
from sklearn.neighbors import KNeighborsRegressor
import numpy as np
import itertools
from math import comb

def makeColors(cat, mag_col_names, use_all_colors=False):
    count = 0
    for i in range(len(mag_col_names) - 1):
        if use_all_colors:
            for j in range(1, len(mag_col_names) - i):
                color = cat[mag_col_names[i+j]] - cat[mag_col_names[i]]
                color_name = "Color" + str(count)
                cat.add_column(color, name=color_name)
                cat[color_name].meta = {"Band1": i+j, 
                                        "Band2": i}
                count += 1
                
        else:
            color = cat[mag_col_names[i+1]] - cat[mag_col_names[i]]
            cat.add_column(color, name="Color" + str(i))
            
def perturbPhotometry(mag, magerr, perturb_in_flux=False, max_mag_error=3, min_flux=1e-35):
    """randomly adjust photometry mags within error bars."""
    rng = np.random.default_rng()
    
    # set maximum magerr
    magerr[magerr > 3] = max_mag_error
    
    if perturb_in_flux:
        flux = 10**(-0.4 * (mag + 48.6))
        snr = 2.5 / np.log(10) / magerr
        noise = flux / snr
        flux += rng.standard_normal(size=len(mag)) * noise
        flux[flux < 0] = min_flux
        newmag = -2.5 * np.log10(flux) - 48.6
    else:
        newmag = mag + rng.standard_normal(size=len(mag)) * magerr
        
    return newmag

def makeCovMatrix(test_cat, err_col_names):
    # FIXME: Full cov matrix is broken. Only using diagonal terms currently
    # I believe the issue is with inversion of the cov matrix
    # Note: ONly diagonal terms were used in Graham et al 2020
    """Construct covariance matrix for Mahalanobis distance.
    # note that while individual magnitudes can be assumed to be independent,
    # colors are not
    # for example, u - g, and g - r both depend on the g band and thus are correlated
    # the covariances can be easily found since in general Cov(X1 + X2, Y) = Cov(X1, Y) + Cov(X2, Y)
    """
    row = test_cat[0]
    keep_columns = []
    for col in list(test_cat.columns):
        if col[0:5] == "Color":
            keep_columns.append(col)
    color_cat = test_cat.copy()
    color_cat.keep_columns(keep_columns)
    
    
    cov = np.zeros((len(keep_columns), len(keep_columns)))
    names = list(color_cat.columns)
    for i in range(cov.shape[0]):
        i_column = names[i]
        
        i_band1 = err_col_names[color_cat[i_column].meta["Band1"]]
        i_band2 = err_col_names[color_cat[i_column].meta["Band2"]]
       
        for j in range(cov.shape[1]):
            # color is Band1 - Band2
            j_column = names[j]
            j_band1 = err_col_names[color_cat[j_column].meta["Band1"]]
            j_band2 = err_col_names[color_cat[j_column].meta["Band2"]]      
          
            #if i_band1 == j_band1:
            #    cov[i, j] += row[i_band1]**2
            #elif  i_band1 == j_band2:
            #    cov[i, j] += -row[i_band1]**2
                
            #if i_band2 == j_band1:
            #    cov[i, j] += -row[i_band2]**2
            #elif i_band2 == j_band2:
            #    cov[i, j] += row[i_band2]**2  
            
            if i_band1 == j_band1 and i_band2 == j_band2:
                cov[i,j] += row[i_band1]**2 + row[i_band2]**2
      
    return cov


def knn_fit(train_cat, test_cat, mag_col_names, y_col_name, err_col_names=[], 
            use_all_colors=False, perturb_training=False, perturb_test=False, 
            n_perturb_test=10, perturb_in_flux=True, metric='euclidean', n_neighbors=10,  
            return_distro=False, return_distance=False, verbose=False):
    """Apply knn regression to determine galaxy property from colors.
    
    Parameters:
    -----------
    
    train_cat :: Astropy Table object
      Table containing data of the training set (i.e. these galaxy colors will create the
      neighborhood space).
      
    test_cat :: Astropy Table object
      Table containing data of galaxies you would like to determine properties for. Each object
      in this table will be compared to the training set to find the nearest neighbors.
      
    mag_col_names :: list of strings
      List of the column names that have the observed magnitudes of the galaxies. Names must be 
      the same in both train_cat and test_cat.
      
    y_col_name :: str
      Column name for the property you want to fit (e.g. 'photoz').
      
    err_col_names :: list of strings
      (Optional) List of column names containing the error (standard deviation) of the observed mags.
      Must be same size as mag_col_names if used. Needed if perturb_training, perturb_test are True,
      or if metric is 'mahalanobis'.
      
    use_all_colors :: Bool
      If True, all possible color combinations will be used (number of mags choose 2). If False,
      colors will be made as if mag_col_names is an ordered list and only adjacent filters will
      be converted to colors. Default False.
      
    perturb_training :: Bool
      If True, create additional realizations of the training set by perturbing the photometry within
      the error bars. This is recommended if using more than one nearest neighbor to ensure additional
      scatter is not introduced by using the median. Default is True.
      
    perturb_test :: Bool
      Create additional realizations of the test set by perturbing the photometry within
      the error bars. Default is False.
      
    n_perturb_test :: int
      Number of different perturbations to perform on test data.
      
    perturb_in_flux :: Bool
      Convert magnitudes and errors into flux space before applying random perturbations. Recommended
      because magnitude measurements are not actually normally distributed. Not necessary if errors 
      are small. Default True.
      
    metric :: str
      Distance metric to use when finding nearest neighbors. Options are 'euclidean' or 'mahalanobis'.
      Using the mahalanobis distance will construct a covariance matrix to weight the distances based
      on the photometric uncertainty. Because of this, only one test object can be fit at a time.
      
    n_neighbors :: int
      Number of nearest neighbors to find
      
    return_distro :: Bool
      If True, will return the entire distribution of fits from all perturbations and nearest neighbors.
      Otherwise, the median will be returned. Default False.
    """
    
    if verbose:
        print("Copying input catalogs")
    train_cat = Table(train_cat, masked=True, copy=True)
    test_cat = Table(test_cat, masked=True, copy=True)
    
    if verbose:
        print("Removing NaNs")
    # remove nans
    for col in mag_col_names:
        train_cat[col].mask = np.isnan(train_cat[col].data) | train_cat[col].mask
        test_cat[col].mask = np.isnan(test_cat[col].data) | test_cat[col].mask
        
        train_cat = train_cat[~train_cat[col].mask] # then remove nans from training set
        test_cat = test_cat[~test_cat[col].mask] # then remove nans from test set
            
    if metric == "mahalanobis":
        if len(test_cat) != 1:
            raise ValueError("Only one object can be fit at a time when using mahalanobis distance. "+ 
                             "Make sure your test_cat has length 1.")
        
    
    
    #perturb things if desired
    
    if perturb_training:
        if verbose:
            print("Perturbing Training Set")
        # copy training cat N times so as to not bias estimate from median
        new_cat = train_cat.copy()
        for i in range(n_neighbors - 1):
            perturbed_cat = train_cat.copy()
            # randomly perturb photometry
            for mag, err in zip(mag_col_names, err_col_names):
                perturbed_cat[mag] = perturbPhotometry(train_cat[mag], train_cat[err], 
                                                       perturb_in_flux=perturb_in_flux)     
            new_cat = vstack((new_cat, perturbed_cat))
        train_cat = new_cat
        
    if perturb_test:
        if verbose:
            print("Perturbing Test Set")
        new_cat = test_cat.copy()
        orig_length = len(test_cat)
        for i in range(n_perturb_test - 1):
            perturbed_cat = test_cat.copy()
            # randomly perturb photometry
            for mag, err in zip(mag_col_names, err_col_names):
                perturbed_cat[mag] = perturbPhotometry(test_cat[mag], test_cat[err], 
                                                       perturb_in_flux=perturb_in_flux)       
            new_cat = vstack((new_cat, perturbed_cat))
        test_cat = new_cat
        
   
        
    if verbose:
            print("Calculating colors")
    # calculate colors
    makeColors(train_cat, mag_col_names, use_all_colors=use_all_colors)
    makeColors(test_cat, mag_col_names, use_all_colors=use_all_colors)
    keep_cols = []
    for n in train_cat.colnames:
        if n[0:5] == "Color":
            keep_cols.append(n)
            
    if metric == "mahalanobis":
        if verbose:
            print("Calculating covariance matrix")
        cov = makeCovMatrix(test_cat, err_col_names)
    
    if verbose:
            print("Converting to color catalog numpy array")
    # convert from fits table to numpy arrays      
    y = train_cat[y_col_name].data
    train_cat.keep_columns(keep_cols)
    X = np.lib.recfunctions.structured_to_unstructured(train_cat.as_array()) 
    X = X.filled(fill_value=np.nan)
    y = y.filled(fill_value=np.nan)

    test_cat.keep_columns(keep_cols)
    X_test = np.lib.recfunctions.structured_to_unstructured(test_cat.as_array())
    X_test = X_test.filled(fill_value=np.nan)
    
    # create tree
    if verbose:
            print("Creating K Neighbours Regressor")
    if metric == "euclidean":
        neigh = KNeighborsRegressor(n_neighbors=n_neighbors)
    elif metric == "mahalanobis":
        
        neigh = KNeighborsRegressor(n_neighbors=n_neighbors, 
                                    metric='mahalanobis', 
                                    metric_params={'VI': np.linalg.inv(cov)})
    else:
        raise ValueError("Metric must be either 'euclidean' or 'mahalanobis'.")
    
    if verbose:
            print("Fitting test data")
    neigh.fit(X, y)
    
    # find nearest neighbor indexes
    dist, ind = neigh.kneighbors(X_test)
    
    if perturb_test:
        temp = ind[0:orig_length]
        temp_dist = dist[0:orig_length]
        for i in range(n_perturb_test):
            temp = np.hstack((temp, ind[orig_length*i:orig_length*(i+1)]))
            temp_dist = np.hstack((temp_dist, dist[orig_length*i:orig_length*(i+1)]))
        ind = temp
        dist = temp_dist
        
    if return_distance:
        if return_distro or n_neighbors == 1:
            return y[ind], dist
        else:
            return np.median(y[ind], axis=1), np.median(dist, axis=1)
    else:
        if return_distro or n_neighbors == 1:
            return y[ind]
        else:
            return np.median(y[ind], axis=1)
    
def getTrueY(test_cat, mag_col_names, y_col_name):

    test_cat = Table(test_cat, masked=True, copy=True)
    # remove nans
    for col in mag_col_names:
        test_cat[col].mask = np.isnan(test_cat[col].data) | test_cat[col].mask
        test_cat = test_cat[~test_cat[col].mask] # then remove nans from test set
            
    true_y = test_cat[y_col_name]
    return true_y.filled()


