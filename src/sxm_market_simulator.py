import json, os, warnings, time, asyncio
import pandas as pd
from pathlib2 import Path
from prefect import task, flow, get_run_logger
from prefect.client.orchestration import get_client
from prefect.blocks.system import Secret
from prefect.context import get_run_context
from prefect.deployments import run_deployment
from prefect.filesystems import Azure
from prefect.task_runners import SequentialTaskRunner
from prefect_ray import RayTaskRunner
from prefect_shell import shell_run_command
from shutil import rmtree
from tqdm import tqdm
from zipfile import ZipFile

results_store = Azure.load("results-store")
global instanceId

from pssh.clients import SSHClient
from pssh.clients.native.single import SFTPIOError
from pssh.exceptions import SessionError
from ssh2.exceptions import SocketRecvError

# import custom library functions
from sxmpy.common import setPandas
from sxmpy.tasks import upload_to_mongodb, parse_config_file
from sxmpy.connections import create_mongodb_query

warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter('ignore', category=FutureWarning)

# ------------------------------------------------------------------------------------------------------------------------------------------

@task
async def unzip_data(zipfile=None, data_dir=None, hostId=None):
    logger = get_run_logger()
    d = {}

    dfMeta = pd.DataFrame(columns=['fpath', 'file_size', 'compress_size'])
    with ZipFile(zipfile, 'r') as zip:
        # for debugging
        for compressed in zip.infolist():
            d['fpath'] = compressed.filename
            d['file_size'] = compressed.file_size
            d['compress_size'] = compressed.compress_size
            dfMeta = pd.concat([dfMeta, pd.Series(d).to_frame().T], ignore_index=True)
        logger.info(f"Compression Metadata:\n{dfMeta}")

    # create data directory if it doesn't exist
    for directory in ['config','book-builder']:
        dirpath = Path(f'/sx3m/data/host{hostId}/{directory}/')
        if not dirpath.exists():
            Path.mkdir(dirpath, parents=True, exist_ok=True)

    # change permissions on data directory
    # await shell_run_command.fn(command=f"chmod -R 777 /sx3m/data/", return_all=True, cwd='/tmp/')

    # extract all files
    cmd = f"""unzip -DD -o -X {zipfile} 'host{hostId}/config/*' 'host{hostId}/book-builder/*' -d {data_dir}/"""
    stdout = await shell_run_command.fn(command=cmd, return_all=True, cwd='/tmp/')

    logger.info(f"stdout: {stdout}")
    # await shell_run_command.fn(command=cmd, return_all=True, cwd='/tmp/')

    return

def remove_file_or_directory(file, strEngineDate):
    if file.name == "data":
        rmfpath = Path('/sx3m', file.name, strEngineDate)
        rmtree(rmfpath.as_posix())
        print(f"removed {rmfpath.as_posix()}")
    else:
        rmfpath = Path('/sx3m', file.name.replace('system', strEngineDate))
        rmfpath.unlink()
        print(f"removed {rmfpath.as_posix()}")


@task(log_prints=True)
async def cleanup_local_files(raw_datapath, configpath, strEngineDate):
    logger = get_run_logger()
    logger.info("PROCESS COMPLETE. CLEANING UP LOCAL FILES..")
    try:
        for file in [raw_datapath, configpath]:
            remove_file_or_directory(file, strEngineDate)
    except Exception as e:
        logger.error(f"ERROR CLEANING UP LOCAL FILES:\n {e}", exc_info=True)


@task
def check_metadata(dir_metadata, metaCollectionName):
    logger = get_run_logger()

    primary_keys = dir_metadata['_id'].unique().tolist()
    logger.info(f"Checking cloud archive status for unique _ids: '{primary_keys}'...")

    market_simulator_meta = create_mongodb_query.fn(dbName='sxmdb',
                                                    collection=metaCollectionName,
                                                    query={'_id': {'$in': primary_keys}})

    if market_simulator_meta is not None:
        logger.debug(f"MongoDB query result:\n{market_simulator_meta}")

        # check if any of the primary keys are already in the cloud
        archive_meta_keys = set(market_simulator_meta['_id'].unique().tolist())
        records_found = [x for x in primary_keys if x in archive_meta_keys]
        records_not_found = [x for x in primary_keys if x not in archive_meta_keys]

        # log records that already exist in the cloud
        if len(records_found) > 0:
            for key in records_found:
                logger.debug(f"{metaCollectionName} for record '{key}' already exists in MongoDB. Skipping record...")

        # if some records do not exist
        if len(records_not_found) > 0:
            for key in records_not_found:
                logger.info(f"{metaCollectionName} for record '{key}' not found in MongoDB. Proceeding..")
            logger.warning(
                f"*** No {metaCollectionName} records returned for {len(records_not_found)} _ids. Proceeding with export pipeline... ***")
            return False
        else:
            logger.info(f"*** All {metaCollectionName} records returned. Moving to next directory for export... ***")
            return True

    # if all records do not exist
    else:
        logger.warning(f"*** No {metaCollectionName} records returned for specified _ids. Continuing with export pipeline... ***")
        return False


@task(name="update system.properties")
async def update_system_properties_file(file_path):
    found = False  # Flag to track if the line was found
    updated_lines = []  # List to store updated lines

    with open(file_path, 'r') as file:
        for line in file:
            if line.strip().startswith('sx3m.jlbh.on.startup'):
                line = 'sx3m.jlbh.on.startup=false\n'
                found = True
            updated_lines.append(line)

    if not found:
        updated_lines.append('sx3m.jlbh.on.startup=false\n')

    with open(file_path, 'w') as file:
        file.writelines(updated_lines)
    return

@task()
async def process_data_for_upload(instanceId, instance, archiveDate, df):
    # id for mongodb unique key field
    df['_id'] = instanceId + ":" + df['icebergId'].astype(str) + ":" + df['eventTime'].astype(str)

    # additional columns to help filter data in queries
    df['hostId'] = instanceId
    df['hostName'] = instance
    df['archiveDate'] = archiveDate

    return


@task(name="create shell script", task_run_name="package: {packageName}")
async def create_shell_script(file_path: str, packageName: str, strArgs: str):
    logger = get_run_logger()
    logger.info(f"Creating shell script at {file_path}")

    with open(file_path, 'w') as file:
        file.write(f"""#!/bin/bash
        # Execute this script with the all-jar in lib/ to run queue viewer.
        JAVA_PROPERTIES=""
        # For Java 17
        JAVA_PROPERTIES="$JAVA_PROPERTIES --enable-preview --add-exports=java.base/jdk.internal.ref=ALL-UNNAMED --add-exports=java.base/sun.nio.ch=ALL-UNNAMED --add-exports=jdk.unsupported/sun.misc=ALL-UNNAMED --add-exports=jdk.compiler/com.sun.tools.javac.file=ALL-UNNAMED --add-opens=jdk.compiler/com.sun.tools.javac=ALL-UNNAMED --add-exports=jdk.compiler/com.sun.tools.javac.file=ALL-UNNAMED --add-opens=java.base/java.lang=ALL-UNNAMED --add-opens=java.base/java.lang.reflect=ALL-UNNAMED --add-opens=java.base/java.io=ALL-UNNAMED --add-opens=java.base/java.util=ALL-UNNAMED"
        # For timezone - comment out to have timestamp in UTC
        JAVA_PROPERTIES="$JAVA_PROPERTIES -DtimestampLongConverters.zoneId=America/Chicago"
        echo "Running: java $JAVA_PROPERTIES -cp 'lib/*' com.stonex.sx3m.{packageName} {strArgs}"
        java $JAVA_PROPERTIES -cp "lib/*" com.stonex.sx3m.{packageName} {strArgs}
        """)
    return


@flow(
    # name=,
    cache_result_in_memory=False,
    task_runner=RayTaskRunner(
        init_kwargs={
            "include_dashboard": False,
        }
    )
)
async def execute_market_simulator(packageName=None, exec_args=None):
    """Execute jvm data export """
    logger = get_run_logger()

    # shell script arguments
    docker_flag = "true"
    data_file_path = exec_args[-3]
    results_file_path = exec_args[-2]
    system_properties_file = exec_args[-1]
    strArgs = f"{docker_flag} {data_file_path} {results_file_path} {system_properties_file}"

    # create shell script to run simulation
    shell_file_path = "/sx3m/market_simulator.sh"
    await create_shell_script(file_path=shell_file_path, packageName=packageName, strArgs=strArgs)

    # command to run in shell to execute
    await shell_run_command(command=f"chmod +x {shell_file_path}", return_all=True, cwd='/sx3m/')
    stdout = await shell_run_command(command=shell_file_path, return_all=True, cwd='/sx3m/')

    logger.info(f"stdout: {stdout}")

    # try to read csv file if exists
    try:
        parsed_outfile = pd.read_csv(results_file_path, compression='gzip', usecols=range(1, 52))
    except (pd.errors.EmptyDataError, Exception) as e:
        logger.warning(f"Returned empty file: {qv_outfile}")
        logger.error(f"Exception: {e}", exc_info=True)
        parsed_outfile = pd.DataFrame()

    Path(results_file_path).unlink()
    return parsed_outfile


def create_meta_df(archive_date=None, instance=None, packageName=None, directoryName=None, uploaded=False, nRows=0):
    dir_metadata = {
        "archive_date": archive_date,
        "instance": instance,
        "packageName": packageName,
        "directoryName": directoryName,
        "uploaded": uploaded,
        "nRows": nRows
    }

    # error catch on naming descrepancy
    if dir_metadata['instance'] == "MET":
        dir_metadata['instance'] = "MET1"
    if dir_metadata['instance'] == "MVP":
        dir_metadata['instance'] = "MVP5"

    dfArchiveMeta = pd.DataFrame.from_dict(dir_metadata, orient='index').T
    dfArchiveMeta['nRows'] = dfArchiveMeta['nRows'].astype(int)

    # concat str key
    dfArchiveMeta["_id"] = (dfArchiveMeta["archive_date"].map(str) + ":" + \
                            dfArchiveMeta['instance'].map(str) + ":" + \
                            dfArchiveMeta['packageName'].map(str)).unique().tolist()
    return dfArchiveMeta


@task()
async def get_archive_directories(startDate=None, endDate=None):
    logger = get_run_logger()

    secret_block = await Secret.load("sshgui-credentials")
    cred = json.loads(secret_block.get())

    client = SSHClient(**cred,
                       num_retries=10,
                       retry_delay=5,
                       keepalive_seconds=10
                       )

    # iterate through directories
    cmd = client.run_command(
        """            
            for dir in /home/sbuser/aurora_share/backup/production/prod-sx3m-*/*/     
            do
                dir=${dir%*/}    # remove the trailing "/"
                echo "${dir}"    # print everything after the final "/"
            done
        """
    )
    lstProdDirs = list(cmd.stdout)
    logger.debug(f"lstProdDirs: {lstProdDirs}")

    # get prod names
    dfDirs = pd.DataFrame(lstProdDirs, columns=['dirs'])
    # remove _old dirs
    dfDirs = dfDirs[~(dfDirs['dirs'].str.contains("_old"))]

    prodNames = dfDirs.apply(lambda x: x['dirs'].rsplit('/', maxsplit=2)[1], axis=1).unique()
    logger.info(f"prodNames: {prodNames}")

    # get archive date from dir path
    dfDirs['archive_date'] = dfDirs['dirs'].str.rsplit('/', n=1, expand=True)[1]

    # filter on start and end dates if passed in
    if startDate is not None:
        dfDirs = dfDirs[dfDirs['archive_date'] >= startDate]
    # if no start date passed in, use the most recent archive date
    else:
        startDate = dfDirs['archive_date'].max()

    logger.info(f"startDate: {startDate}")
    dfDirs = dfDirs[dfDirs['archive_date'] >= startDate]

    if endDate is not None:
        logger.info(f"endDate: {endDate}")
        dfDirs = dfDirs[dfDirs['archive_date'] <= endDate]

    dfDirs.sort_values(['archive_date', 'dirs'], ascending=False, inplace=True)
    logger.debug(f"Directory meta:\n{dfDirs}")

    # todo: skip ice for now (due to iceberg issues)
    dfDirs['engineName'] = dfDirs['dirs'].str.rsplit('/', n=2, expand=True)[1].str.rsplit('-', n=1, expand=True)[1].str.upper()
    dfDirs = dfDirs.query("~engineName.isin(['IFUS1','IFUS2'])")

    client.disconnect()
    return dfDirs.groupby('archive_date').agg({'engineName':'unique','dirs':'unique'})


@flow(name="Parallel Archive Market Simulator",
    flow_run_name="{dataDirectory} | {packageName}",
    task_runner=RayTaskRunner(
        init_kwargs={
        "include_dashboard": False,
        # "dashboard_host": "0.0.0.0",
        # "dashboard_port": 8265,
        # "num_cpus": 1,
        # "num_gpus": 0,
        # "memory": "1GB",
        # "object_store_memory": "1GB",

    }),
    # result_storage=,
    # result_serializer=,
    # persist_result=,
    cache_result_in_memory=False,
    # log_prints=True,
)
async def ParallelArchiveMarketSimulator(dataDirectory=None, packageName=None):
    logger = get_run_logger()
    secret_block = await Secret.load("sshgui-credentials")
    cred = json.loads(secret_block.get())

    # db and metaDb names for simulation:
    dataCollectionName = f'mktSim_{packageName}'
    metaCollectionName = f'mktSim_meta'

    client = SSHClient(**cred,
                       num_retries=10,
                       retry_delay=5,
                       keepalive_seconds=10
                       )

    dctValidate = {}
    dfMeta = create_meta_df(
        archive_date=dataDirectory.rsplit('/', maxsplit=1)[1],
        instance=dataDirectory.rsplit('/')[-2].rsplit('-', 1)[1].upper(),
        packageName=packageName,
        directoryName=dataDirectory.rsplit('/', maxsplit=2)[1].lower(),
        uploaded=False,
        nRows=0
    )
    archiveDate, instance = dfMeta['archive_date'][0], dfMeta['instance'][0]

    # filter based on change to spreads in August 2023
    if archiveDate >= "20230801" and instance not in ["MET1", "MVP5", "ENG1"]:
        logger.info(f"Skipping {archiveDate} for {instance} due to change in spreads data")
        return
    if archiveDate <= "20230801" and instance not in ["MET1", "SPD1", "ENG1"]:
        logger.info(f"Skipping {archiveDate} for {instance} due to change in spreads data")
        return

    # check if archive has already been uploaded
    if check_metadata.submit(dfMeta, metaCollectionName).result():
        pass
    else:
        configpath = Path(dataDirectory, "system.properties")
        compressed_datapath = Path(dataDirectory, "data.zip")
        raw_datapath = Path(dataDirectory, "data")

        # create specific name
        strEngineDate = f"{dfMeta['instance'][0]}_{dfMeta['archive_date'][0]}"

        # try to catch disconnect error
        try:
            # check if all files exist / if data is available
            for fpath in [configpath, compressed_datapath, raw_datapath]:
                cmd = client.run_command(f"test -e {fpath.as_posix()} && echo 1 || echo 0")
                dctValidate[fpath.name] = int(list(cmd.stdout)[0])

            # if uncompressed data folder exists and other files needed exist, zip data folder
            if dctValidate['data'] == 1 and dctValidate['data.zip'] == 0 and dctValidate['system.properties'] == 1:
                logger.warning(f"{compressed_datapath.as_posix()} not found. Compressing {raw_datapath.as_posix()} to {compressed_datapath.as_posix()}")

                # zip data folder
                cmd = client.run_command(f"zip -r {compressed_datapath.as_posix()} {raw_datapath.as_posix()}")
                # remove data folder
                cmd = client.run_command(f"rm -rf {raw_datapath.as_posix()}")
                # set data.zip to 1
                dctValidate['data.zip'] = 1

            # if all files exist, copy to local machine
            if dctValidate['data.zip'] == 1 and dctValidate['system.properties'] == 1:
                logger.info("All files exist. Beginning copy to local machine..")

                for file in [configpath, compressed_datapath]:
                    try:
                        logger.info(f"Copying {file.as_posix()}")

                        # get host id and name from system.properties
                        if file.name == 'system.properties':
                            fpath = Path('/sx3m', file.name.replace('system', strEngineDate)).as_posix()
                            client.copy_remote_file(file.as_posix(), fpath)

                            # get hostId from system.properties file
                            await update_system_properties_file(file_path=fpath)
                            parser = parse_config_file.submit(fpath=fpath).result()
                            instanceId = parser.get('default', 'instance.hostId')

                            continue

                        if file.name == "data.zip" and file.exists():
                            # check if volume is mounted to container directory
                            logger.info("aurora_share volume is mounted to the container: *** Skipping COPY-FILE ***")

                            fpath = file.as_posix()

                            logger.info(f"Unzipping {fpath} to /sx3m/data/{strEngineDate}/")
                            await unzip_data(zipfile=fpath,
                                                    data_dir=f"/sx3m/data/{strEngineDate}/",
                                                    hostId=instanceId,
                                                    )
                        else:
                            logger.warning("aurora_share volume is NOT mounted to the container: *** Using SSHFS to copy file ***")

                            fpath = Path('/tmp', file.name.replace("data", f"{strEngineDate}")).as_posix()
                            client.copy_remote_file(file.as_posix(), fpath)
                            logger.info(f"Copying {file.as_posix()} to {fpath}")

                            await unzip_data(zipfile=fpath,
                                                    data_dir=f"/sx3m/data/{strEngineDate}/",
                                                    hostId=instanceId,
                                                    # wait_for=[parse_config_file],
                                                    )

                            # Check if the top-level path is '/tmp'
                            if Path(fpath).exists() and Path(fpath).parent.as_posix() == '/tmp':
                                # Check if the file ends with '.zip'
                                if fpath.endswith('.zip'):
                                    # Remove the file
                                    logger.info(f"Removing: {fpath}")
                                    Path(fpath).unlink()
                                else:
                                    logger.info(f"File does not end with .zip: {fpath}")
                            else:
                                logger.warning(f"File is not in /tmp: {fpath}")

                    except SFTPIOError as e:
                        logger.error(f"SFTP ERROR COPYING FILE: {file}")

                try:
                    dfResults = await execute_market_simulator.with_options(
                        flow_run_name=f"{dfMeta['instance'][0]}/{dfMeta['archive_date'][0]} {packageName}",
                    )(packageName=f"{packageName}",
                      exec_args=[
                          "true",
                          f"/data/{strEngineDate}/",
                          f"/tmp/results_{strEngineDate}.csv.gz",
                          f"/sx3m/{strEngineDate}.properties"
                      ],
                      wait_for=[unzip_data])

                    if not dfResults.empty:
                        dfUpload = await process_data_for_upload(instanceId=instanceId,
                                                                 instance=instance,
                                                                 archiveDate=archiveDate,
                                                                 df=dfResults,
                                                                 wait_for=[execute_market_simulator])
                        upload_to_mongodb.submit(df=dfUpload, collName=dataCollectionName).result()

                        dfMeta['nRows'] = dfUpload.shape[0]
                        dfMeta['uploaded'] = True
                        upload_to_mongodb.submit(df=dfMeta, collName=metaCollectionName).result()
                    else:
                        logger.warning(f"No messages were exported for {dfMeta['archive_date'].unique()[0]}")

                        dfMeta['nRows'] = 0
                        dfMeta['uploaded'] = True
                        logger.debug(f"dfMeta:\n{dfMeta}")
                        upload_to_mongodb.submit(df=dfMeta, collName=metaCollectionName).result()

                except Exception as e:
                    logger.error(f"ERROR:\n {e}", exc_info=True)

                finally:
                    await cleanup_local_files.submit(raw_datapath=raw_datapath,
                                               configpath=configpath,
                                               strEngineDate=strEngineDate,
                                               wait_for=[execute_market_simulator, upload_to_mongodb])
            else:
                logger.warning(f"WARNING: {dataDirectory} does not have all required files. Skipping...")
        except (SessionError, SocketRecvError, BaseException) as e:
            logger.error(f"ERROR: {e}\n\n", exc_info=True)
    return


@task()
async def poll_flow_run_status(flow_run_id=None):
    logger = get_run_logger()

    while True:
        async with get_client() as client:
            flow_run = await client.read_flow_run(flow_run_id)
            if flow_run.state.is_final():
                break
            logger.info("Flows are still running. Sleeping for 10 seconds...")
            await asyncio.sleep(10)
    return


@flow()
async def run_deployments_by_archive_date(dfArchive=None, packageName=None):
    logger = get_run_logger()

    # Loop through each archive_date in the DataFrame
    for archive_date, row in tqdm(dfArchive.iterrows()):
        engine_names = row['engineName']
        directories = row['dirs']
        flow_run_ids = []

        # Run deployments for each engineName and collect flow_run_ids
        for engine_name, directory in zip(engine_names, directories):
            logger.info(f"Launching {packageName.rsplit('.')[-1]} for {archive_date} and engine {engine_name}")

            parameters = {
                'dataDirectory': directory,
                'packageName': packageName,
            }

            flow_run = await run_deployment(
                name="Parallel Archive Market Simulator/adhoc",
                parameters=parameters,
                timeout=0
            )
            flow_run_ids.append(flow_run.id)
            await asyncio.sleep(10)

        # Wait for the last engineName to complete its flow run
        last_flow_run_id = flow_run_ids[-1]
        await poll_flow_run_status(flow_run_id=last_flow_run_id)
        logger.info(f"f{archive_date} completed successfully. Continuing to next archive_date...")

    return


@flow(name="Market Simulator",
    flow_run_name="{packageName} | {startDate} - {endDate}",
      # task_runner=SequentialTaskRunner(),
)
async def MarketSimulator(packageName=None, exec_args=None, startDate=None, endDate=None):
    # get and filter directories to export
    dfDirs = await get_archive_directories(startDate=startDate, endDate=endDate)
    # parallel run deployments
    await run_deployments_by_archive_date(dfArchive=dfDirs, packageName=packageName)


#####################################################
# for local debugging

if __name__ == "__main__":
    setPandas(4)
    # asyncio.run(MarketSimulator(packageName="analytics.icebergsimulation.IcebergSimulation", exec_args=None, startDate="20230623", endDate="20230705", dropData=""))
    asyncio.run(ParallelArchiveMarketSimulator(dataDirectory="/home/sbuser/aurora_share/backup/production/prod-sx3m-spd1/20230623", packageName="analytics.icebergsimulation.IcebergSimulation"))
