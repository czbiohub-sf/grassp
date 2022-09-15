
import argparse

from some_package.some_subpackage import some_module


def parse_args(argv):

    parser = argparse.ArgumentParser()

    # a sample id
    parser.add_argument('--sample-id', dest='sample_id', type=str, required=True)

    # an example of a binary flag (here used to indicate whether to run in a 'verbose' mode)
    parser.add_argument(
        '--verbose', dest='verbose', action='store_true', required=False, default=False
    )

    args = parser.parse_args(argv.split(' ') if argv else None)
    return args


def main(argv=None):
    '''
    A demo CLI interface that calls the method `some_package.some_subpackage.some_module.some_method`
    It has a required --sample-id argument and an optional --verbose flag.

    The command-line alias 'some-cli' is defined in setup.py to point to this method.

    Example usage:
    ```
    some-cli --sample-id sample01 --verbose
    ```
    This generates the following output:
    ```
    Processing sample PML0123
    some_method was called with sample_id 'sample01'
    Finished processing sample PML0123
    ```

    If the --verbose flag is omitted, then the output is simply
    ```
    some_method was called with sample_id 'sample01'
    ```
    '''

    args = parse_args(argv)

    if args.verbose:
        print('Processing sample %s' % args.sample_id)
        
    some_module.some_method(args.sample_id)
    
    if args.verbose:
        print('Finished processing sample %s' % args.sample_id)


if __name__ == '__main__':
    main()
