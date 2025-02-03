import datetime
import numpy as np

def run_experiments(
    title,
    dataset_name,
    experiments,
    ds_dir,
    fs_dir,
    user_ids,
    options={},
    setup=None,
):
    start_ts = datetime.datetime.now()

    print_line_break()
    print_title(title)

    if callable(setup):
        print_line_break()
        setup()

    results = []
    total_experiments = np.size(experiments)

    for i, experiment in enumerate(experiments):
        experiment_start_ts = datetime.datetime.now()
        experiment_id = i + 1
        
        print_line_break()
        print_message(
            'Running experiment %d of %d...' % (
                experiment_id,
                total_experiments,
        ))

        result = experiment(
            dataset_name=dataset_name,
            ds_dir=ds_dir,
            fs_dir=fs_dir,
            user_ids=user_ids,
            options=options,
        )
        results.append(result)

        experiment_end_ts = datetime.datetime.now()

        print_line_break()
        print_message('Time elapsed in experiment %d of %d: %s' % (
            experiment_id,
            total_experiments,
            str(experiment_end_ts - experiment_start_ts),
        ))

    end_ts = datetime.datetime.now()

    print_line_break()
    print_message('Finished all experiments')
    print_message('Total time elapsed: %s' % str(end_ts - start_ts))
    print_line_break()
    print_result('# Experimental results')

    for result in results:
        if 'message' in result:
            print_line_break()
            print_result(result['message'])

    print_line_break()

    return results

def get_formatted_time():
    now = datetime.datetime.now()
    return '{YEAR:04d}-{MONTH:02d}-{DAY:02d} {HOUR:02d}:{MINUTE:02d}:{SECOND:02d}'.format(
        YEAR=now.year,
        MONTH=now.month,
        DAY=now.day,
        HOUR=now.hour,
        MINUTE=now.minute,
        SECOND=now.second,
    )

def print_message(message):
    print('{DATE} {MESSAGE}'.format(
        DATE=get_formatted_time(),
        MESSAGE=message,
    ))

def print_title(message):
    print(message)

def print_result(message):
    print(message)

def print_line_break():
    print()

def print_progress(task, progress, status):
    print('{DATE} [{PROGRESS:5.1f}%] {TASK}: {STATUS}'.format(
        DATE=get_formatted_time(),
        TASK=task,
        PROGRESS=progress*100,
        STATUS=status,
    ))

def download_assets(asset_manager, assets, dir=None):
    start_ts = datetime.datetime.now()
    
    task = 'Download HGR datasets'
    total_files =  len(assets.keys())

    print_progress(
        task,
        progress=0.0,
        status='downloading %d files...' % total_files,
    )

    for i, key in enumerate(assets.keys()):
        asset_manager.add_remote_asset(
            key,
            assets[key]['remote_id'],
            assets[key]['filename'],
        )

        cached = asset_manager.download_asset(key, dir)

        if cached:
            message = 'found file %2d of %2d in local cache (%s)'
        else:
            message = 'downloaded file %2d of %2d (%s)' 

        print_progress(
            task,
            progress=(i+1)/total_files,
            status=message % (
                i+1,
                total_files,
                assets[key]['filename'],
            ),
        )
    
    end_ts = datetime.datetime.now()
    print_line_break()
    print_message('Finished downloading files')
    print_message('Time elapsed downloading files: %s' % str(end_ts - start_ts))

    return asset_manager
