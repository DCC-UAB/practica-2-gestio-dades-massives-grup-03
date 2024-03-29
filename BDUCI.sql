DROP TABLE "DATASET" cascade constraints;
DROP TABLE "SAMPLES" cascade constraints;
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
) SEGMENT CREATION IMMEDIATE
  PCTFREE 10 PCTUSED 40 INITRANS 1 MAXTRANS 255 NOCOMPRESS LOGGING
  STORAGE(INITIAL 65536 NEXT 1048576 MINEXTENTS 1 MAXEXTENTS 2147483645
  PCTINCREASE 0 FREELISTS 1 FREELIST GROUPS 1 BUFFER_POOL DEFAULT FLASH_CACHE DEFAULT CELL_FLASH_CACHE DEFAULT)
  TABLESPACE "USERS" ;

CREATE UNIQUE INDEX "CLASSIFICADOR_PK" ON "CLASSIFICADOR" ("NOMCURT")
    PCTFREE 10 INITRANS 2 MAXTRANS 255 COMPUTE STATISTICS
  STORAGE(INITIAL 65536 NEXT 1048576 MINEXTENTS 1 MAXEXTENTS 2147483645
  PCTINCREASE 0 FREELISTS 1 FREELIST GROUPS 1 BUFFER_POOL DEFAULT FLASH_CACHE DEFAULT CELL_FLASH_CACHE DEFAULT)
  TABLESPACE "USERS" ;
ALTER TABLE "CLASSIFICADOR" ADD CONSTRAINT "CLASSIFICADOR_PK" PRIMARY KEY ("NOMCURT");
ALTER TABLE "CLASSIFICADOR" MODIFY ("NOMCURT" NOT NULL ENABLE);

-- Creación de la tabla "PARAMETRES" con dos columnas, una de las cuales almacena datos JSON como CLOB
CREATE TABLE "PARAMETRES"
(
    "NOMCURT" VARCHAR2(20),
    "HASH" VARCHAR2(64),
    "VALORS" JSON -- Se asume que se almacenan datos JSON como CLOB
) SEGMENT CREATION IMMEDIATE
  PCTFREE 10 PCTUSED 40 INITRANS 1 MAXTRANS 255 NOCOMPRESS LOGGING
  STORAGE(INITIAL 65536 NEXT 1048576 MINEXTENTS 1 MAXEXTENTS 2147483645
  PCTINCREASE 0 FREELISTS 1 FREELIST GROUPS 1 BUFFER_POOL DEFAULT FLASH_CACHE DEFAULT CELL_FLASH_CACHE DEFAULT)
  TABLESPACE "USERS" ;

ALTER TABLE "PARAMETRES" ADD CONSTRAINT "PARAMETRES_PK" PRIMARY KEY ("HASH", "NOMCURT");
ALTER TABLE "PARAMETRES" MODIFY ("NOMCURT" NOT NULL ENABLE);
ALTER TABLE "PARAMETRES" MODIFY ("HASH" NOT NULL ENABLE);
ALTER TABLE "PARAMETRES" ADD CONSTRAINT "PARAMETRES_FK" FOREIGN KEY ("NOMCURT")
    REFERENCES "CLASSIFICADOR" ("NOMCURT") ON DELETE CASCADE ENABLE;

-- Creación de la tabla "REPETICIO" con dos columnas y una restricción de clave primaria en "NAMEDATASET"
CREATE TABLE "REPETICIO"
(
    "NAMEDATASET" VARCHAR2(20),
    "NUM" NUMBER
)SEGMENT CREATION IMMEDIATE
  PCTFREE 10 PCTUSED 40 INITRANS 1 MAXTRANS 255 NOCOMPRESS LOGGING
  STORAGE(INITIAL 65536 NEXT 1048576 MINEXTENTS 1 MAXEXTENTS 2147483645
  PCTINCREASE 0 FREELISTS 1 FREELIST GROUPS 1 BUFFER_POOL DEFAULT FLASH_CACHE DEFAULT CELL_FLASH_CACHE DEFAULT)
  TABLESPACE "USERS"  ;


-- Creación de la tabla "EXPERIMENT" con cinco columnas
CREATE TABLE "EXPERIMENT"
(
    "NAMEDATASET" VARCHAR2(20),
    "NOMCURT" VARCHAR2(20),
    "PAR_HASH" VARCHAR2(64),
    "DATA" DATE,
    "ACCURACY" NUMBER,
    "F_SCORE" NUMBER
)SEGMENT CREATION IMMEDIATE
  PCTFREE 10 PCTUSED 40 INITRANS 1 MAXTRANS 255 NOCOMPRESS LOGGING
  STORAGE(INITIAL 65536 NEXT 1048576 MINEXTENTS 1 MAXEXTENTS 2147483645
  PCTINCREASE 0 FREELISTS 1 FREELIST GROUPS 1 BUFFER_POOL DEFAULT FLASH_CACHE DEFAULT CELL_FLASH_CACHE DEFAULT)
  TABLESPACE "USERS"  ;

ALTER TABLE "EXPERIMENT" MODIFY ("NOMCURT" NOT NULL ENABLE);

-- Se agregan restricciones de clave foránea en la tabla "EXPERIMENT" haciendo referencia a las tablas "REPETICIO" y "CLASSIFICADOR"
ALTER TABLE "EXPERIMENT" ADD CONSTRAINT "EXPERIMENT_FK_DATASET" FOREIGN KEY ("NAMEDATASET")
REFERENCES "DATASET" ("NAME") ON DELETE CASCADE ENABLE;
ALTER TABLE "EXPERIMENT" ADD CONSTRAINT "EXPERIMENT_FK_CLASSIFICADOR" FOREIGN KEY ("PAR_HASH", "NOMCURT")
REFERENCES "PARAMETRES" ("HASH", "NOMCURT") ON DELETE CASCADE ENABLE;

GRANT SELECT ON GestorUCI.DATASET TO TestUCI;
GRANT SELECT ON GestorUCI.REPETICIO TO TestUCI;
GRANT SELECT ON GestorUCI.PARAMETRES TO TestUCI;
GRANT SELECT ON GestorUCI.CLASSIFICADOR TO TestUCI;

GRANT INSERT ON GestorUCI.REPETICIO TO TestUCI;
GRANT UPDATE ON GestorUCI.REPETICIO TO TestUCI;

GRANT INSERT ON GestorUCI.EXPERIMENT TO TestUCI;
GRANT UPDATE ON GestorUCI.EXPERIMENT TO TestUCI;

GRANT INSERT ON GestorUCI.REPETICIO TO DevUCI;
GRANT UPDATE ON GestorUCI.REPETICIO TO DevUCI;
GRANT DELETE ON GestorUCI.REPETICIO TO DevUCI;

GRANT INSERT ON GestorUCI.EXPERIMENT TO DevUCI;
GRANT UPDATE ON GestorUCI.EXPERIMENT TO DevUCI;
GRANT DELETE ON GestorUCI.EXPERIMENT TO DevUCI;

GRANT INSERT ON GestorUCI.PARAMETRES TO DevUCI;
GRANT UPDATE ON GestorUCI.PARAMETRES TO DevUCI;
GRANT DELETE ON GestorUCI.PARAMETRES TO DevUCI;

GRANT INSERT ON GestorUCI.CLASSIFICADOR TO DevUCI;
GRANT UPDATE ON GestorUCI.CLASSIFICADOR TO DevUCI;
GRANT DELETE ON GestorUCI.CLASSIFICADOR TO DevUCI;