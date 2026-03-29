import argparse
import json


def main(nested_dict):
    # Example: Process the nested dictionary
    print(nested_dict)


def parse_nested_dict(nested_dict, is_json_str=False):
    # Convert json str into a deeply nested dict of dicts, lists, functions and their outputs.
    # Note that we can copy over values from other parts of the json `before` they are processed. That is anything
    # that depends on randomness better incorporate some sort of random_seed fixing + may take extra time.
    # Parse JSON string into dictionary
    nested_dict = json.loads(nested_dict) if is_json_str else nested_dict
    # Copy over any values that need to be copied
    # Looping allows to cover cases where a copied value contains another copy.
    # Avoid recursive copying - this has not been tested and will probably not work properly.
    for i in range(5):
        nested_dict = preprocess_value(nested_dict, nested_dict)
    # Convert everything as described in `parse_objects` docstring.
    nested_dict = parse_objects(nested_dict)
    return nested_dict


def parse_objects(obj):
    # Parse functions and objects from strings.
    # Custom functions:
    # {"__func__": "lambda x: x+1"} -> <function <lambda>>
    # {"__func__": "lambda x: x+1", "__args__": {x: 2}} -> 3
    # Predefined functions, classes and class instances:
    # {"__class__": "min"} -> <built-in function min>
    # {"__class__": "rand", "__module__": "numpy.random"} -> <built-in method rand of numpy.random.mtrand.Randomstate object>
    # {"__class__": "ones", "__module__": "numpy", "__args__": [1,2]} -> array([1., 1.])
    # Does not support combining args and kwargs.
    # To make `__func__` work with imported functions the following can be used:
    # {"__func__": "lambda f, x: f(x)", "__args__": [{"__module__": module_name, "__class__": fn_name}, x]}
    # becomes module_name.fn_name(x).
    # `__func__` might even work with imported functions but not sure
    # nor relying on that - instead just use `__class__` with `__module__`.
    if "__func__" in obj:
        # Evaluate function
        func_str = obj["__func__"]
        ret_obj = eval(func_str)
        if "__args__" in obj:
            args = parse_value(obj.get("__args__", {}))
            ret_obj = ret_obj(**args) if isinstance(args, dict) else ret_obj(*args)
        return ret_obj

    elif "__class__" in obj:
        module_name = obj.get("__module__", "builtins")
        module = __import__(module_name, fromlist=[obj["__class__"]])
        ret_obj = getattr(module, obj["__class__"])
        if "__args__" in obj:
            args = parse_value(obj.get("__args__"))
            ret_obj = ret_obj(**args) if isinstance(args, dict) else ret_obj(*args)
        return ret_obj
    return {k: parse_value(v) for k, v in obj.items()}


def preprocess_value(value, root):
    # Recursively iterate through a nested dict/list structure.
    # Replace {"__copy__": a.b.c} with root[a][b][c].
    # "a.2.c" can successfully index {"a": [{},{"c":True}]} or {"a": {2: {"c": True}}}.
    # To get to {"a": {"2": {"c": True}}} use "a.'2'.c".
    if isinstance(value, dict):
        if "__copy__" in value:
            new_value = root
            for k in value["__copy__"].split("."):
                try:
                    k = int(k)
                except ValueError:
                    pass
                new_value = new_value[k]
            return new_value
        else:
            return {k: preprocess_value(v, root) for k, v in value.items()}
    elif isinstance(value, list):
        return [preprocess_value(v, root) for v in value]
    return value


def parse_value(value):
    # Recursively parse values
    if isinstance(value, dict):
        return parse_objects(value)
    elif isinstance(value, list):
        return [parse_value(item) for item in value]
    else:
        return value


def parse_single_sweep_run_config(
    d, rm_wandb=False, rm_profiler=False, rm_checkpointing=False
):
    # Use this to convert the config of a run from a sweep to config usable stand-alone.
    # Basically wandb adds "desc" and "value" outer dicts, which we remove.
    # Also to avoid instantiating a wandb run can optionally remove the wandb param and some objects that tend
    # to copy the run name for path purposes.
    if rm_wandb and isinstance(d, dict) and "wandb" in d:
        d.pop("wandb")
        rm_wandb = False
    if rm_profiler and isinstance(d, dict) and "profiler" in d:
        d.pop("profiler")
        rm_profiler = False
    if rm_checkpointing and isinstance(d, dict) and "checkpointing" in d:
        d.pop("checkpointing")
        rm_checkpointing = False
    if isinstance(d, dict) and "desc" in d and "value" in d:
        return parse_single_sweep_run_config(
            d["value"], rm_wandb, rm_profiler, rm_checkpointing
        )
    elif isinstance(d, dict):
        return {
            k: parse_single_sweep_run_config(v, rm_wandb, rm_profiler, rm_checkpointing)
            for k, v in d.items()
        }
    elif isinstance(d, list):
        return [
            parse_single_sweep_run_config(v, rm_wandb, rm_profiler, rm_checkpointing)
            for v in d
        ]
    else:
        return d


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process deeply nested dictionary")
    parser.add_argument(
        "--nested_dict", type=str, help="Deeply nested dictionary as a JSON string"
    )
    args = parser.parse_args()

    if args.nested_dict:
        try:
            nested_dict = parse_nested_dict(args.nested_dict)
            main(nested_dict)
        except json.JSONDecodeError:
            print("Invalid JSON string provided.")
    else:
        print("No nested dictionary provided.")
