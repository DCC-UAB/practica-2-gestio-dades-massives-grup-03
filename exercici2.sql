DROP TABLE "DATASET" cascade constraints;
DROP TABLE "SAMPLES" cascade constraints;
DROP TABLE "REPETICIO" cascade constraints;
DROP TABLE "EXPERIMENT" cascade constraints;
DROP TABLE "PARAMETRES" cascade constraints;
DROP TABLE "CLASSIFICADOR" cascade constraints;
--------------------------------------------------------
--  DDL for Table DATASET
--------------------------------------------------------

  CREATE TABLE "DATASET"
   (	"NAME" VARCHAR2(20),
    "FEAT_SIZE" NUMBER,
	"NUMCLASSES" NUMBER,
	"INFO" JSON
   ) SEGMENT CREATION IMMEDIATE
  PCTFREE 10 PCTUSED 40 INITRANS 1 MAXTRANS 255 NOCOMPRESS LOGGING
  STORAGE(INITIAL 65536 NEXT 1048576 MINEXTENTS 1 MAXEXTENTS 2147483645
  PCTINCREASE 0 FREELISTS 1 FREELIST GROUPS 1 BUFFER_POOL DEFAULT FLASH_CACHE DEFAULT CELL_FLASH_CACHE DEFAULT)
  TABLESPACE "USERS"  ;

--------------------------------------------------------
--  DDL for Table FEATUREVECTOR
--------------------------------------------------------

  CREATE TABLE "SAMPLES"
   (	"NAMEDATASET" VARCHAR2(20),
	"ID" NUMBER,
	"FEATURES" BLOB,
	"LABEL" VARCHAR2(16)
   ) SEGMENT CREATION IMMEDIATE
  PCTFREE 10 PCTUSED 40 INITRANS 1 MAXTRANS 255 NOCOMPRESS LOGGING
  STORAGE(INITIAL 65536 NEXT 1048576 MINEXTENTS 1 MAXEXTENTS 2147483645
  PCTINCREASE 0 FREELISTS 1 FREELIST GROUPS 1 BUFFER_POOL DEFAULT FLASH_CACHE DEFAULT CELL_FLASH_CACHE DEFAULT)
  TABLESPACE "USERS"
 LOB ("FEATURES") STORE AS BASICFILE (
  TABLESPACE "USERS" ENABLE STORAGE IN ROW CHUNK 8192 RETENTION
  NOCACHE LOGGING
  STORAGE(INITIAL 65536 NEXT 1048576 MINEXTENTS 1 MAXEXTENTS 2147483645
  PCTINCREASE 0 FREELISTS 1 FREELIST GROUPS 1 BUFFER_POOL DEFAULT FLASH_CACHE DEFAULT CELL_FLASH_CACHE DEFAULT)) ;

--------------------------------------------------------
--  DDL for Index DATASET_PK
--------------------------------------------------------

  CREATE UNIQUE INDEX "DATASET_PK" ON "DATASET" ("NAME")
  PCTFREE 10 INITRANS 2 MAXTRANS 255 COMPUTE STATISTICS
  STORAGE(INITIAL 65536 NEXT 1048576 MINEXTENTS 1 MAXEXTENTS 2147483645
  PCTINCREASE 0 FREELISTS 1 FREELIST GROUPS 1 BUFFER_POOL DEFAULT FLASH_CACHE DEFAULT CELL_FLASH_CACHE DEFAULT)
  TABLESPACE "USERS" ;
--------------------------------------------------------
--  DDL for Index SAMPLES_PK
--------------------------------------------------------

  CREATE UNIQUE INDEX "SAMPLES_PK" ON "SAMPLES" ("ID", "NAMEDATASET")
  PCTFREE 10 INITRANS 2 MAXTRANS 255 COMPUTE STATISTICS
  STORAGE(INITIAL 65536 NEXT 1048576 MINEXTENTS 1 MAXEXTENTS 2147483645
  PCTINCREASE 0 FREELISTS 1 FREELIST GROUPS 1 BUFFER_POOL DEFAULT FLASH_CACHE DEFAULT CELL_FLASH_CACHE DEFAULT)
  TABLESPACE "USERS" ;

--------------------------------------------------------
--  Constraints for Table DATASET
--------------------------------------------------------

  ALTER TABLE "DATASET" ADD CONSTRAINT "DATASET_PK" PRIMARY KEY ("NAME")
  USING INDEX PCTFREE 10 INITRANS 2 MAXTRANS 255 COMPUTE STATISTICS
  STORAGE(INITIAL 65536 NEXT 1048576 MINEXTENTS 1 MAXEXTENTS 2147483645
  PCTINCREASE 0 FREELISTS 1 FREELIST GROUPS 1 BUFFER_POOL DEFAULT FLASH_CACHE DEFAULT CELL_FLASH_CACHE DEFAULT)
  TABLESPACE "USERS"  ENABLE;

  ALTER TABLE "DATASET" MODIFY ("NAME" NOT NULL ENABLE);

--------------------------------------------------------
--  Constraints for Table SAMPLES
--------------------------------------------------------

  ALTER TABLE "SAMPLES" ADD CONSTRAINT "SAMPLES_PK" PRIMARY KEY ("ID", "NAMEDATASET")
  USING INDEX PCTFREE 10 INITRANS 2 MAXTRANS 255 COMPUTE STATISTICS
  STORAGE(INITIAL 65536 NEXT 1048576 MINEXTENTS 1 MAXEXTENTS 2147483645
  PCTINCREASE 0 FREELISTS 1 FREELIST GROUPS 1 BUFFER_POOL DEFAULT FLASH_CACHE DEFAULT CELL_FLASH_CACHE DEFAULT)
  TABLESPACE "USERS"  ENABLE;

  ALTER TABLE "SAMPLES" MODIFY ("NAMEDATASET" NOT NULL ENABLE);

  ALTER TABLE "SAMPLES" MODIFY ("ID" NOT NULL ENABLE);


--------------------------------------------------------
--  Ref Constraints for Table SAMPLES
--------------------------------------------------------

ALTER TABLE "SAMPLES" ADD CONSTRAINT "SAMPLES_FK" FOREIGN KEY ("NAMEDATASET")
  REFERENCES "DATASET" ("NAME") ON DELETE CASCADE ENABLE;

-- Creación de la tabla "CLASSIFICADOR" con dos columnas y una restricción de clave primaria
CREATE TABLE "CLASSIFICADOR"
(
    "NOMCURT" VARCHAR2(20),
    "NOM" VARCHAR2(100)
);

CREATE UNIQUE INDEX "CLASSIFICADOR_PK" ON "CLASSIFICADOR" ("NOMCURT");
ALTER TABLE "CLASSIFICADOR" ADD CONSTRAINT "CLASSIFICADOR_PK" PRIMARY KEY ("NOMCURT");
ALTER TABLE "CLASSIFICADOR" MODIFY ("NOMCURT" NOT NULL ENABLE);

-- Creación de la tabla "PARAMETRES" con dos columnas, una de las cuales almacena datos JSON como CLOB
CREATE TABLE "PARAMETRES"
(
    "NOMCURT" VARCHAR2(20),
    "HASH" VARCHAR2(64),
    "VALORS" JSON -- Se asume que se almacenan datos JSON como CLOB
);

ALTER TABLE "PARAMETRES" ADD CONSTRAINT "PARAMETRES_PK" PRIMARY KEY ("HASH", "NOMCURT");
ALTER TABLE "PARAMETRES" MODIFY ("NOMCURT" NOT NULL ENABLE);
ALTER TABLE "PARAMETRES" MODIFY ("HASH" NOT NULL ENABLE);
ALTER TABLE "PARAMETRES" MODIFY ("VALORS" NOT NULL ENABLE);
ALTER TABLE "PARAMETRES" ADD CONSTRAINT "PARAMETRES_FK" FOREIGN KEY ("NOMCURT")
    REFERENCES "CLASSIFICADOR" ("NOMCURT") ON DELETE CASCADE ENABLE;

-- Creación de la tabla "REPETICIO" con dos columnas y una restricción de clave primaria en "NAMEDATASET"
CREATE TABLE "REPETICIO"
(
    "NAMEDATASET" VARCHAR2(20),
    "NUM" NUMBER
);


ALTER TABLE "REPETICIO" ADD CONSTRAINT "REPETICIO_PK" PRIMARY KEY ("NUM", "NAMEDATASET");

ALTER TABLE "REPETICIO" MODIFY ("NAMEDATASET" NOT NULL ENABLE);

ALTER TABLE "REPETICIO" MODIFY ("NUM" NOT NULL ENABLE);

ALTER TABLE "REPETICIO" ADD CONSTRAINT "REPETICIO_FK" FOREIGN KEY ("NAMEDATASET")
  REFERENCES "DATASET" ("NAME") ON DELETE CASCADE ENABLE;

-- Creación de la tabla "EXPERIMENT" con cinco columnas
CREATE TABLE "EXPERIMENT"
(
    "NAMEDATASET" VARCHAR2(20),
    "NOMCURT" VARCHAR2(20),
    "DATA" DATE,
    "ACCURACY" NUMBER,
    "F_SCORE" NUMBER
);

ALTER TABLE "EXPERIMENT" MODIFY ("NOMCURT" NOT NULL ENABLE);

-- Se agregan restricciones de clave foránea en la tabla "EXPERIMENT" haciendo referencia a las tablas "REPETICIO" y "CLASSIFICADOR"
ALTER TABLE "EXPERIMENT" ADD CONSTRAINT "EXPERIMENT_FK_DATASET" FOREIGN KEY ("NAMEDATASET")
REFERENCES "DATASET" ("NAME") ON DELETE CASCADE ENABLE;

ALTER TABLE "EXPERIMENT" ADD CONSTRAINT "EXPERIMENT_FK_CLASSIFICADOR" FOREIGN KEY ("NOMCURT")
REFERENCES "CLASSIFICADOR" ("NOMCURT") ON DELETE CASCADE ENABLE;