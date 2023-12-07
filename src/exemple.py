def insertVectorDataset(dbConn, nameDataset, fileName, label_pos, *args, **kwargs):
  # Leer los datos del archivo y generar un dataframe
  df, _ = readVectorDataFile(fileName, label_pos=label_pos)

  # Iniciar la conexión con la base de datos
  cursor = dbConn.cursor()

  # Verificar si el dataset ya existe
  cursor.execute("SELECT COUNT(*) FROM Dataset WHERE NAME = :1", [nameDataset])
  yaExiste = cursor.fetchone()[0] != 0

  if not yaExiste:
    # Calcular tamaño de las características
    tamanoCaracteristicas = len(df['features'].iloc[0])

    # Determinar la posición de la etiqueta y calcular el número de clases
    etiqueta_pos = 1 if label_pos != -1 else -1
    numeroClases = len(df.iloc[:, etiqueta_pos].unique())

    # Preparar y ejecutar la inserción de los datos del dataset
    cursor.prepare("INSERT INTO Dataset (NAME, FEAT_SIZE, NUMCLASSES) VALUES (:1, :2, :3)")
    cursor.execute(None, [nameDataset, tamanoCaracteristicas, numeroClases])
    logging.warning(
      f"Insertando dataset {nameDataset} con {tamanoCaracteristicas} características y {numeroClases} clases")

    # Preparar la consulta para insertar muestras
    cursor.prepare("INSERT INTO Samples (NAMEDATASET, ID, FEATURES, LABEL) VALUES (:1, :2, :3, :4)")
    for index, row in df.iterrows():
      blobCaracteristicas = cursor.var(oracledb.BLOB)
      blobCaracteristicas.setvalue(0, np.array(row['features']).tobytes())
      etiqueta = row.iloc[etiqueta_pos]
      cursor.execute(None, [nameDataset, index + 1, blobCaracteristicas, etiqueta])

    mensaje_commit = "Datos insertados correctamente en el nuevo dataset."
  else:
    mensaje_commit = f"El dataset {nameDataset} ya existe, no se realizaron inserciones."

  # Intentar hacer commit y manejar excepciones
  try:
    dbConn.commit()
    print(mensaje_commit)
    return True
  except Exception as e:
    print(f"Error al hacer commit: {e}")
    return False