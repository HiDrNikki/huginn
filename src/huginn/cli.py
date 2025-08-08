import argparse
from pathlib import Path
from dotenv import load_dotenv
import os
load_dotenv()

def buildParser(cliTemplates):
    parser = argparse.ArgumentParser(prog="test")
    subParsers = parser.add_subparsers(dest="command", required=True)
    for template in cliTemplates:
        cmd = template["command"]
        description = template.get("description", None)
        func = template.get("func", None)
        args = template.get("args", [])
        sub = subParsers.add_parser(cmd, help=description)
        for arg in args:
            flags = arg.get("flags", [])
            kwargs = {k: v for k, v in arg.items() if k not in ("name", "flags")}
            if flags:
                sub.add_argument(*flags, dest=arg["name"], **kwargs)
            else:
                sub.add_argument(arg["name"], **kwargs)
        if func:
            sub.set_defaults(func=func)
    return parser

def helpHandler(args):
    parser = args._parser
    if args.command:
        actions = [a for a in parser._actions if isinstance(a, argparse._SubParsersAction)]
        if actions:
            subparser = actions[0].choices.get(args.command)
            if subparser:
                subparser.print_help()
            else:
                parser.print_help()
        else:
            parser.print_help()
    else:
        parser.print_help()

def testHandler(args):
    try:
        from huginn.aiAssistant import AIAssistant

        assistant = AIAssistant(
            hfToken=os.getenv("hfToken"),
            modelID="google/gemma-2b",
            quantized=False,
            cache=Path("D:/huggingface")
        )
        
        outputText = assistant.generate("Hello")
    except Exception as error:
        print("Test failed:")
        print(f"    {error}")
        return
    
    print(outputText)

cliTemplates = [
    {
        "command": "test",
        "description": "Create an assistant that says hello.",
        "func": testHandler,
    },
    {
        "command": "help",
        "description": "Show help for a command.",
        "func": helpHandler,
        "args": [
            {
                "name": "command",
                "flags": ["-c", "--command"],
                "nargs": "?",
                "help": "Show help for this command."
            }
        ]
    }
]

def main():
    parser = buildParser(cliTemplates)
    args = parser.parse_args()
    # Attach parser for use in help
    setattr(args, "_parser", parser)
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()