import h5py


path = "outputs/matriz_sim_general.h5"

with h5py.File(path, "r") as f:
    print("\nDatasets en el archivo:", list(f.keys()))

    for k in f.keys():
        ds = f[k]
        print(f"\n➡ Dataset: {k}")

        print("   - Tipo:", type(ds))
        print("   - Shape:", ds.shape)
        print("   - Dtype:", ds.dtype)

    # Si existe dataset 'cvegeo'
    if "cvegeo" in f:
        print("\nEjemplo cvegeo:", f["cvegeo"][:5])

    # Si existe un dataset llamado 'matriz'
    if "matriz" in f:
        m = f["matriz"]
        print("\nPrimeros 5x5 valores de la matriz:")
        print(m[:5, :5])
