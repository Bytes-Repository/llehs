use comfy_table::{Table, presets::UTF8_FULL};
use std::collections::{HashMap, VecDeque};
use std::env;
use std::fs::{self, File};
use std::io::{self, BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio, Child, ChildStdin, ChildStdout};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use colored::Colorize;
use directories::ProjectDirs;
use git2::Repository;
use miette::{Diagnostic, Result as MietteResult};
use nix::sys::wait::{waitpid, WaitStatus};
use nix::unistd::{fork, ForkResult, Pid};
use notify_rust::Notification;
use rayon::prelude::*;
use reedline::{
    default_emacs_keybindings, DefaultHighlighter, FileBackedHistory,
    KeyCode, KeyModifiers, Prompt, PromptEditMode, PromptHistorySearch, Reedline, ReedlineEvent, Signal as ReedlineSignal,
    PromptViMode,
};
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;
use shellexpand::tilde;
use signal_hook::iterator::Signals;
use wasmtime::{Engine, Linker, Module, Store};
use which::which_all;

#[derive(Debug, Clone, Serialize, Deserialize)]
enum LlehsValue {
    String(String),
    Int(i64),
    Float(f64),
    Bool(bool),
    List(Vec<LlehsValue>),
    Table(HashMap<String, LlehsValue>),
    Duration(Duration),
    Filesize(u64),
    Error(String),
}

impl LlehsValue {
    fn to_json(&self) -> JsonValue {
        match self {
            LlehsValue::String(s) => JsonValue::String(s.clone()),
            LlehsValue::Int(i) => JsonValue::Number((*i).into()),
            LlehsValue::Float(f) => JsonValue::Number(serde_json::Number::from_f64(*f).unwrap()),
            LlehsValue::Bool(b) => JsonValue::Bool(*b),
            LlehsValue::List(l) => JsonValue::Array(l.iter().map(|v| v.to_json()).collect()),
            LlehsValue::Table(t) => JsonValue::Object(t.iter().map(|(k, v)| (k.clone(), v.to_json())).collect()),
            LlehsValue::Duration(d) => JsonValue::String(format!("{:?}", d)),
            LlehsValue::Filesize(s) => JsonValue::Number((*s).into()),
            LlehsValue::Error(e) => JsonValue::String(format!("Error: {}", e)),
        }
    }

    fn from_json(json: JsonValue) -> Self {
        match json {
            JsonValue::String(s) => LlehsValue::String(s),
            JsonValue::Number(n) => if n.is_i64() {
                LlehsValue::Int(n.as_i64().unwrap())
            } else {
                LlehsValue::Float(n.as_f64().unwrap())
            },
            JsonValue::Bool(b) => LlehsValue::Bool(b),
            JsonValue::Array(a) => LlehsValue::List(a.into_iter().map(Self::from_json).collect()),
            JsonValue::Object(o) => LlehsValue::Table(o.into_iter().map(|(k, v)| (k, Self::from_json(v))).collect()),
            _ => LlehsValue::Error("Unsupported JSON type".to_string()),
        }
    }

    fn length(&self) -> LlehsValue {
        match self {
            LlehsValue::List(l) => LlehsValue::Int(l.len() as i64),
            _ => LlehsValue::Int(0),
        }
    }
}

#[derive(Debug, Clone)]
struct CommandContext {
    current_dir: PathBuf,
    env_vars: HashMap<String, String>,
    sandbox: bool,
    dry_run: bool,
    variables: HashMap<String, LlehsValue>,
    last_exit_code: i32,
}

#[derive(Debug)]
struct BackgroundJob {
    pid: Pid,
    command: String,
}

struct LlehsEngine {
    plugins: HashMap<String, Module>,
    engine: Engine,
    jobs: Arc<Mutex<VecDeque<BackgroundJob>>>,
    history: Vec<String>,
}

impl LlehsEngine {
    fn new() -> MietteResult<Self> {
        let engine = Engine::default();
        Ok(Self {
            plugins: HashMap::new(),
            engine,
            jobs: Arc::new(Mutex::new(VecDeque::new())),
            history: Vec::new(),
        })
    }

    fn load_plugin(&mut self, name: &str, path: &Path) -> MietteResult<()> {
        let wasm_bytes = fs::read(path).map_err(|e| miette::miette!("Failed to read plugin: {}", e))?;
        let module = Module::new(&self.engine, &wasm_bytes).map_err(|e| miette::miette!("Failed to compile WASM: {}", e))?;
        self.plugins.insert(name.to_string(), module);
        Ok(())
    }

    fn execute_plugin(&self, name: &str, input: &LlehsValue) -> MietteResult<LlehsValue> {
        if let Some(module) = self.plugins.get(name) {
            let mut linker = Linker::new(&self.engine);
            linker.define("host", "input", wasmtime::ExternRef::new(input.to_json()))?;
            let mut store = Store::new(&self.engine, ());
            let instance = linker.instantiate(&mut store, module)?;
            let func = instance.get_func(&mut store, "run").ok_or(miette::miette!("No 'run' function in plugin"))?;
            let mut results = vec![wasmtime::Val::ExternRef(wasmtime::ExternRef::null())];
            func.call(&mut store, &[], &mut results)?;
            if let wasmtime::Val::ExternRef(ref_) = &results[0] {
                if let Some(json) = ref_.data().downcast_ref::<JsonValue>() {
                    return Ok(LlehsValue::from_json(json.clone()));
                }
            }
            Err(miette::miette!("Plugin execution failed"))
        } else {
            Err(miette::miette!("Plugin not found: {}", name))
        }
    }
}

#[derive(Clone)]
struct LlehsPrompt {
    context: Arc<Mutex<CommandContext>>,
}

impl Prompt for LlehsPrompt {
    fn render_prompt_left(&self) -> String {
        let ctx = self.context.lock().unwrap();
        let dir = ctx.current_dir.to_string_lossy();
        let short_dir = if dir.starts_with(&env::var("HOME").unwrap_or_default()) {
            dir.replacen(&env::var("HOME").unwrap(), "~", 1)
        } else {
            dir.to_string()
        };

        let status = if ctx.last_exit_code == 0 {
            "✔".green()
        } else {
            "✘".red()
        };

        let branch = get_git_branch(&ctx.current_dir).unwrap_or_default();
        let branch_str = if branch.is_empty() { "".to_string() } else { format!(" ({})", branch.magenta()) };

        format!("{} {} {}", status, short_dir.blue(), branch_str)
    }

    fn render_prompt_right(&self) -> String {
        "".to_string()
    }

    fn render_prompt_indicator(&self, _edit_mode: PromptEditMode) -> String {
        "> ".to_string()
    }

    fn render_prompt_vi_indicator(&self, _vi_mode: PromptViMode) -> String {
        "".to_string()
    }

    fn render_prompt_multiline_indicator(&self) -> String {
        "::: ".to_string()
    }

    fn render_prompt_history_search_indicator(&self, _history_search: PromptHistorySearch) -> String {
        "".to_string()
    }
}

fn get_git_branch(dir: &Path) -> Option<String> {
    if let Ok(repo) = Repository::open(dir) {
        if let Ok(head) = repo.head() {
            if let Some(name) = head.shorthand() {
                return Some(name.to_string());
            }
        }
    }
    None
}

fn main() -> MietteResult<()> {
    let mut signals = Signals::new([signal_hook::consts::SIGINT])?;
    let signal_thread = thread::spawn(move || {
        for sig in signals.forever() {
            if sig == signal_hook::consts::SIGINT {
                // Handle Ctrl+C
            }
        }
    });

    let dirs = ProjectDirs::from("com", "xAI", "llehs").unwrap();
    let history_path = dirs.data_dir().join("history.txt");
    let mut editor = Reedline::create();
    let history = FileBackedHistory::with_file(1000, history_path.clone()).map_err(|e| miette::miette!("History error: {}", e))?;
    editor = editor.with_history(Box::new(history));

    // Advanced completer with context
    let mut completer = LlehsCompleter::new()?;
    editor = editor.with_completer(Box::new(completer));

    let mut keybindings = default_emacs_keybindings();
    keybindings.add_binding(
        KeyModifiers::NONE,
        KeyCode::Tab,
        ReedlineEvent::UntilFound(vec![ReedlineEvent::Menu("completion_menu".to_string()), ReedlineEvent::MenuNext]),
    );
    editor = editor.with_keybindings(keybindings);

    let mut engine = LlehsEngine::new()?;
    let context = CommandContext {
        current_dir: env::current_dir()?,
        env_vars: env::vars().collect(),
        sandbox: false,
        dry_run: false,
        variables: HashMap::new(),
        last_exit_code: 0,
    };
    let context_arc = Arc::new(Mutex::new(context));
    let prompt = LlehsPrompt { context: context_arc.clone() };

    // Job manager thread
    let jobs_clone = engine.jobs.clone();
    thread::spawn(move || {
        loop {
            {
                let mut jobs = jobs_clone.lock().unwrap();
                jobs.retain(|job| {
                    if let Ok(status) = waitpid(job.pid, nix::sys::wait::WaitPidFlag::WNOHANG) {
                        if matches!(status, WaitStatus::Exited(_, _) | WaitStatus::Signaled(_, _, _)) {
                            Notification::new().summary("Llehs Job Finished").body(&job.command).show().unwrap();
                            false
                        } else {
                            true
                        }
                    } else {
                        false
                    }
                });
            }
            thread::sleep(Duration::from_secs(1));
        }
    });

    loop {
        let sig = editor.read_line(&prompt);
        match sig {
            Ok(ReedlineSignal::Success(line)) => {
                if line.trim().is_empty() {
                    continue;
                }
                let mut ctx = context_arc.lock().unwrap();
                if let Err(e) = process_line(&line, &mut engine, &mut ctx) {
                    eprintln!("{:?}", e);
                    ctx.last_exit_code = 1;
                } else {
                    ctx.last_exit_code = 0;
                }
            }
            Ok(ReedlineSignal::CtrlD) | Ok(ReedlineSignal::CtrlC) => {
                println!("Exiting llehs...");
                break;
            }
            Err(err) => {
                println!("Error: {:?}", err);
                break;
            }
        }
    }

    signal_thread.join().unwrap();
    Ok(())
}

fn process_line(line: &str, engine: &mut LlehsEngine, context: &mut CommandContext) -> MietteResult<LlehsValue> {
    // Simple scripting parser
    if line.starts_with("if ") {
        return parse_if(line, engine, context);
    } else if line.starts_with('$') && line.contains('=') {
        return parse_assignment(line, engine, context);
    }

    let parts: Vec<&str> = line.split('|').map(|s| s.trim()).collect();
    if parts.iter().all(|part| part.split_whitespace().next().unwrap_or("").starts_with('^')) {
        // All external, use system pipes
        return execute_external_pipeline(&parts, context);
    }

    let mut input = LlehsValue::List(vec![]);

    for part in parts {
        let args: Vec<&str> = part.split_whitespace().collect();
        if args.is_empty() {
            continue;
        }

        let mut cmd = args[0];
        let parallel = cmd.ends_with("p");
        if parallel {
            cmd = &cmd[..cmd.len() - 1];
        }

        let is_external = cmd.starts_with('^');
        if is_external {
            cmd = &cmd[1..];
        }

        input = if parallel {
            execute_parallel(cmd, &args[1..], input, context, is_external)?
        } else {
            let mut temp_input = input.clone();
            if !is_external && matches!(cmd, "where" | "length") && matches!(temp_input, LlehsValue::String(_)) {
                // Smart conversion
                if let Ok(parsed) = from_ssv(&temp_input) {
                    temp_input = parsed;
                } else if let Ok(json) = serde_json::from_str::<JsonValue>(&if let LlehsValue::String(s) = temp_input { s } else { unreachable!() }) {
                    temp_input = LlehsValue::from_json(json);
                }
            }
            execute_command(cmd, &args[1..], temp_input, engine, context, is_external)?
        };
    }

    print_value(&input);
    Ok(input)
}

fn execute_external_pipeline(parts: &[&str], context: &CommandContext) -> MietteResult<LlehsValue> {
    let mut children: Vec<Child> = vec![];
    let mut prev_stdout: Option<ChildStdout> = None;

    for (i, part) in parts.iter().enumerate() {
        let args: Vec<&str> = part.split_whitespace().collect();
        let cmd = &args[0][1..]; // Remove ^

        let mut command = Command::new(cmd);
        command.args(&args[1..]);
        command.current_dir(&context.current_dir);
        command.envs(&context.env_vars);

        if i == 0 {
            command.stdin(Stdio::inherit());
        } else {
            command.stdin(Stdio::from(prev_stdout.take().unwrap()));
        }

        if i == parts.len() - 1 {
            command.stdout(Stdio::inherit());
        } else {
            command.stdout(Stdio::piped());
        }
        command.stderr(Stdio::inherit());

        let mut child = command.spawn().map_err(|e| miette::miette!("Spawn failed: {}", e))?;
        prev_stdout = child.stdout.take();
        children.push(child);
    }

    let mut last_status = 0;
    for mut child in children {
        let status = child.wait().map_err(|e| miette::miette!("Wait failed: {}", e))?;
        last_status = status.code().unwrap_or(1);
    }

    if last_status == 0 {
        Ok(LlehsValue::Int(0))
    } else {
        Err(miette::miette!("Pipeline failed with status {}", last_status))
    }
}

fn parse_assignment(line: &str, engine: &mut LlehsEngine, context: &mut CommandContext) -> MietteResult<LlehsValue> {
    let parts: Vec<&str> = line.splitn(2, '=').map(|s| s.trim()).collect();
    if parts.len() != 2 {
        return Err(miette::miette!("Invalid assignment"));
    }
    let var_name = parts[0].strip_prefix('$').unwrap_or(parts[0]);
    let value = process_line(parts[1], engine, context)?;
    context.variables.insert(var_name.to_string(), value.clone());
    Ok(value)
}

fn parse_if(line: &str, engine: &mut LlehsEngine, context: &mut CommandContext) -> MietteResult<LlehsValue> {
    let re = Regex::new(r"if \((.*?)\) \{(.*?)\}").unwrap();
    if let Some(caps) = re.captures(line) {
        let cond_str = caps.get(1).unwrap().as_str();
        let body_str = caps.get(2).unwrap().as_str();

        let cond_with_vars = replace_variables(cond_str, context);
        let cond_result = process_line(&cond_with_vars, engine, context)?;

        let cond_bool = match cond_result {
            LlehsValue::Int(i) if i > 0 => true,
            LlehsValue::Bool(b) => b,
            _ => false,
        };

        if cond_bool {
            return process_line(body_str, engine, context);
        }
        Ok(LlehsValue::String("Condition false".to_string()))
    } else {
        Err(miette::miette!("Invalid if syntax"))
    }
}

fn replace_variables(s: &str, context: &CommandContext) -> String {
    let re = Regex::new(r"\$([a-zA-Z_]+)").unwrap();
    re.replace_all(s, |caps: &regex::Captures| {
        let var = &caps[1];
        if let Some(val) = context.variables.get(var) {
            format!("{:?}", val)
        } else {
            caps[0].to_string()
        }
    }).to_string()
}

fn execute_parallel(cmd: &str, args: &[&str], input: LlehsValue, context: &CommandContext, is_external: bool) -> MietteResult<LlehsValue> {
    if let LlehsValue::List(list) = input {
        let results: Vec<LlehsValue> = list.par_iter().map(|item| {
            execute_command(cmd, args, item.clone(), &mut LlehsEngine::new().unwrap(), context, is_external).unwrap_or(LlehsValue::Error("Parallel exec failed".to_string()))
        }).collect();
        Ok(LlehsValue::List(results))
    } else {
        Err(miette::miette!("Parallel requires list input"))
    }
}

fn execute_command(cmd: &str, args: &[&str], input: LlehsValue, engine: &mut LlehsEngine, context: &CommandContext, is_external: bool) -> MietteResult<LlehsValue> {
    if context.sandbox && !is_safe_command(cmd, context) {
        return Err(miette::miette!("Command blocked in sandbox: {}", cmd));
    }

    if context.dry_run {
        println!("Dry run: Would execute {} with args {:?} on input {:?}", cmd, args, input);
        return Ok(LlehsValue::String("Dry run".to_string()));
    }

    if is_external || matches!(cmd, "curl" | "bg") {
        let full_args = [cmd].iter().chain(args.iter()).cloned().collect::<Vec<&str>>();
        let mut result = execute_external(&full_args, context, Some(&input))?;
        // Auto-parse if possible
        if cmd == "ls" && args.contains(&"-la") {
            result = from_ssv(&result)?;
        } else if cmd == "ps" && args.contains(&"aux") {
            result = from_ssv_ps(&result)?;
        }
        return Ok(result);
    }

    match cmd {
        "ls" => {
            let path = if args.is_empty() { context.current_dir.as_path() } else { Path::new(args[0]) };
            let entries = fs::read_dir(path).map_err(|e| miette::miette!("ls failed: {}", e))?;
            let mut list = vec![];
            for entry in entries {
                let entry = entry?;
                let metadata = entry.metadata()?;
                let size = LlehsValue::Filesize(metadata.len());
                let name = LlehsValue::String(entry.file_name().to_string_lossy().to_string());
                list.push(LlehsValue::Table(HashMap::from([("name".to_string(), name), ("size".to_string(), size)])));
            }
            Ok(LlehsValue::List(list))
        }
        "cd" => {
            let path_str = if args.is_empty() { "~" } else { args[0] };
            let path = PathBuf::from(tilde(path_str).to_string());
            env::set_current_dir(&path).map_err(|e| miette::miette!("cd failed: {}", e))?;
            Ok(LlehsValue::String("OK".to_string()))
        }
        "exit" => std::process::exit(0),
        "load_plugin" => {
            if args.len() < 2 {
                return Err(miette::miette!("Usage: load_plugin <name> <path>"));
            }
            engine.load_plugin(args[0], Path::new(args[1]))?;
            Ok(LlehsValue::String("Plugin loaded".to_string()))
        }
        "run_plugin" => {
            if args.is_empty() {
                return Err(miette::miette!("Usage: run_plugin <name>"));
            }
            engine.execute_plugin(args[0], &input)
        }
        "where" => {
            if args.len() < 3 {
                return Err(miette::miette!("Usage: where <field> <op> <value>"));
            }
            if let LlehsValue::List(list) = input {
                let field = args[0];
                let op = args[1];
                let val = parse_value(args[2]);
                let filtered: Vec<LlehsValue> = list.into_iter().filter(|item| {
                    if let LlehsValue::Table(t) = item {
                        if let Some(v) = t.get(field) {
                            compare_values(v, op, &val)
                        } else {
                            false
                        }
                    } else {
                        false
                    }
                }).collect();
                Ok(LlehsValue::List(filtered))
            } else {
                Err(miette::miette!("Where requires list input"))
            }
        }
        "from-ssv" => from_ssv(&input),
        "length" => Ok(input.length()),
        "notify" => {
            Notification::new().summary("Llehs Notification").body(args[0]).show().unwrap();
            Ok(LlehsValue::String("Notified".to_string()))
        }
        "bg" => {
            if args.is_empty() {
                return Err(miette::miette!("Usage: bg <command>"));
            }
            let full_cmd = args.join(" ");
            match unsafe { fork() } {
                Ok(ForkResult::Parent { child }) => {
                    engine.jobs.lock().unwrap().push_back(BackgroundJob { pid: child, command: full_cmd.clone() });
                    Ok(LlehsValue::String(format!("Job started: {}", child)))
                }
                Ok(ForkResult::Child) => {
                    let full_args = full_cmd.split_whitespace().collect::<Vec<&str>>();
                    execute_external(&full_args, context, None)?;
                    std::process::exit(0);
                }
                Err(e) => Err(miette::miette!("Fork failed: {}", e)),
            }
        }
        "curl" => {
            if args.len() < 1 {
                return Err(miette::miette!("Usage: curl <url>"));
            }
            if args[0].contains("| sh") || args[0].contains("| bash") {
                println!("Warning: Executing piped curl from internet is dangerous!");
            }
            execute_external(&[cmd, args[0]], context, None)
        }
        _ => Err(miette::miette!("Unknown command: {}", cmd)),
    }
}

fn from_ssv(input: &LlehsValue) -> MietteResult<LlehsValue> {
    if let LlehsValue::String(s) = input {
        let lines: Vec<&str> = s.lines().collect();
        let mut list = vec![];
        for line in lines {
            let cols: Vec<String> = line.split_whitespace().map(|c| c.to_string()).collect();
            if cols.len() >= 9 { // For ls -la format
                let mut table = HashMap::new();
                table.insert("perms".to_string(), LlehsValue::String(cols[0].clone()));
                table.insert("links".to_string(), LlehsValue::String(cols[1].clone()));
                table.insert("owner".to_string(), LlehsValue::String(cols[2].clone()));
                table.insert("group".to_string(), LlehsValue::String(cols[3].clone()));
                table.insert("size".to_string(), parse_value(&cols[4]));
                table.insert("date".to_string(), LlehsValue::String(cols[5..8].join(" ")));
                table.insert("name".to_string(), LlehsValue::String(cols[8..].join(" ")));
                list.push(LlehsValue::Table(table));
            }
        }
        Ok(LlehsValue::List(list))
    } else {
        Err(miette::miette!("from-ssv requires string input"))
    }
}

fn from_ssv_ps(input: &LlehsValue) -> MietteResult<LlehsValue> {
    if let LlehsValue::String(s) = input {
        let lines: Vec<&str> = s.lines().skip(1).collect(); // Skip header
        let mut list = vec![];
        for line in lines {
            let cols: Vec<String> = line.split_whitespace().map(|c| c.to_string()).collect();
            if cols.len() >= 11 {
                let mut table = HashMap::new();
                table.insert("user".to_string(), LlehsValue::String(cols[0].clone()));
                table.insert("pid".to_string(), LlehsValue::String(cols[1].clone()));
                table.insert("cpu".to_string(), LlehsValue::String(cols[2].clone()));
                table.insert("mem".to_string(), LlehsValue::String(cols[3].clone()));
                table.insert("vsz".to_string(), LlehsValue::String(cols[4].clone()));
                table.insert("rss".to_string(), LlehsValue::String(cols[5].clone()));
                table.insert("tty".to_string(), LlehsValue::String(cols[6].clone()));
                table.insert("stat".to_string(), LlehsValue::String(cols[7].clone()));
                table.insert("start".to_string(), LlehsValue::String(cols[8].clone()));
                table.insert("time".to_string(), LlehsValue::String(cols[9].clone()));
                table.insert("command".to_string(), LlehsValue::String(cols[10..].join(" ")));
                list.push(LlehsValue::Table(table));
            }
        }
        Ok(LlehsValue::List(list))
    } else {
        Err(miette::miette!("from-ssv-ps requires string input"))
    }
}

fn execute_external(args: &[&str], context: &CommandContext, input: Option<&LlehsValue>) -> MietteResult<LlehsValue> {
    let mut command = Command::new(args[0]);
    command.args(&args[1..]);
    command.current_dir(&context.current_dir);
    command.envs(&context.env_vars);
    command.stderr(Stdio::inherit());

    if let Some(LlehsValue::String(s)) = input {
        let mut child = command
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()
            .map_err(|e| miette::miette!("Spawn failed: {}", e))?;
        {
            let stdin = child.stdin.as_mut().unwrap();
            stdin.write_all(s.as_bytes()).map_err(|e| miette::miette!("Write to stdin failed: {}", e))?;
        }
        let output = child.wait_with_output().map_err(|e| miette::miette!("Wait failed: {}", e))?;
        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        if output.status.success() {
            Ok(LlehsValue::String(stdout))
        } else {
            Err(miette::miette!("Command failed with status {}", output.status))
        }
    } else {
        let mut child = command
            .stdin(Stdio::inherit())
            .stdout(Stdio::inherit())
            .spawn()
            .map_err(|e| miette::miette!("Nie znaleziono komendy '{}': {}", args[0], e))?;
        let status = child.wait().map_err(|e| miette::miette!("Błąd czekania na proces: {}", e))?;

        if status.success() {
            Ok(LlehsValue::Int(status.code().unwrap_or(0) as i64))
        } else {
            Err(miette::miette!("Command failed with status {}", status))
        }
    }
}

fn is_safe_command(_cmd: &str, _context: &CommandContext) -> bool {
    true // Implement properly
}

fn compare_values(a: &LlehsValue, op: &str, b: &LlehsValue) -> bool {
    match op {
        ">" => match (a, b) {
            (LlehsValue::Filesize(s1), LlehsValue::Filesize(s2)) => s1 > s2,
            (LlehsValue::Int(i1), LlehsValue::Int(i2)) => i1 > i2,
            _ => false,
        },
        "==" => match (a, b) {
            (LlehsValue::String(s1), LlehsValue::String(s2)) => s1 == s2,
            _ => false,
        },
        // Add more
        _ => false,
    }
}

fn parse_value(s: &str) -> LlehsValue {
    if let Ok(i) = s.parse::<i64>() {
        LlehsValue::Int(i)
    } else if s.ends_with("mb") {
        if let Ok(mb) = s[..s.len()-2].parse::<u64>() {
            LlehsValue::Filesize(mb * 1024 * 1024)
        } else {
            LlehsValue::String(s.to_string())
        }
    } else if s.ends_with("gb") {
        if let Ok(gb) = s[..s.len()-2].parse::<u64>() {
            LlehsValue::Filesize(gb * 1024 * 1024 * 1024)
        } else {
            LlehsValue::String(s.to_string())
        }
    } else {
        LlehsValue::String(s.to_string())
    }
}

fn print_value(value: &LlehsValue) {
    match value {
        LlehsValue::List(list) if !list.is_empty() && matches!(list[0], LlehsValue::Table(_)) => {
            let mut table = Table::new();
            table.set_style(UTF8_FULL);

            // Assume all tables have same keys
            if let LlehsValue::Table(first) = &list[0] {
                let headers: Vec<String> = first.keys().cloned().collect();
                table.set_header(headers.clone());
                for item in list {
                    if let LlehsValue::Table(t) = item {
                        let row: Vec<String> = headers.iter().map(|k| format!("{:?}", t.get(k).unwrap_or(&LlehsValue::String("".to_string())))).collect();
                        table.add_row(row);
                    }
                }
                println!("{}", table);
            }
        }
        LlehsValue::List(list) => {
            for item in list {
                print_value(item);
            }
        }
        LlehsValue::Table(table) => {
            for (k, v) in table {
                println!("{}: {:?}", k, v);
            }
        }
        _ => println!("{:?}", value),
    }
}

#[derive(Clone)]
struct LlehsCompleter {
    commands: Vec<String>,
    binaries: Vec<String>,
    history: Vec<String>,
}

impl LlehsCompleter {
    fn new() -> MietteResult<Self> {
        let mut commands = vec!["ls".to_string(), "cd".to_string(), "exit".to_string(), "where".to_string(), "load_plugin".to_string(), "run_plugin".to_string(), "bg".to_string(), "from-ssv".to_string(), "length".to_string(), "notify".to_string()];
        let mut binaries = vec![];

        if let Ok(path) = env::var("PATH") {
            for dir in path.split(':') {
                if let Ok(entries) = fs::read_dir(dir) {
                    for entry in entries.flatten() {
                        if entry.file_type().map(|ft| ft.is_file() || ft.is_symlink()).unwrap_or(false) {
                            binaries.push(entry.file_name().to_string_lossy().to_string());
                        }
                    }
                }
            }
        }

        commands.extend(binaries.iter().cloned());
        commands.sort();
        commands.dedup();

        Ok(Self {
            commands,
            binaries,
            history: vec![],
        })
    }
}

impl reedline::Completer for LlehsCompleter {
    fn complete(&mut self, line: &str, pos: usize) -> Vec<reedline::Suggestion> {
        let mut suggestions = vec![];
        let last_word = line.split_whitespace().last().unwrap_or(line);

        if line.starts_with("cd ") {
            let prefix = &line[3..pos];
            if let Ok(entries) = fs::read_dir(".") {
                for entry in entries.flatten() {
                    let name = entry.file_name().to_string_lossy().to_string();
                    if name.starts_with(prefix) {
                        suggestions.push(reedline::Suggestion {
                            value: name,
                            description: None,
                            extra: None,
                            span: reedline::Span { start: line.len() - prefix.len(), end: pos },
                            append_whitespace: false,
                        });
                    }
                }
            }
        } else {
            for cmd in &self.commands {
                if cmd.starts_with(last_word) {
                    suggestions.push(reedline::Suggestion {
                        value: cmd.clone(),
                        description: Some(format!("Command: {}", cmd)),
                        extra: None,
                        span: reedline::Span { start: line.len() - last_word.len(), end: pos },
                        append_whitespace: true,
                    });
                }
            }
        }
        // Add flag suggestions, etc.
        if line.contains("tar ") {
            suggestions.push(reedline::Suggestion {
                value: "-xvf".to_string(),
                description: Some("Extract verbose file".to_string()),
                extra: None,
                span: reedline::Span { start: line.len(), end: line.len() },
                append_whitespace: false,
            });
        }
        // Frequent paths
        if line.starts_with("cd p") {
            suggestions.insert(0, reedline::Suggestion {
                value: "~/projects/rust".to_string(),
                description: Some("Frequent path".to_string()),
                extra: None,
                span: reedline::Span { start: 3, end: pos },
                append_whitespace: false,
            });
        }
        suggestions
    }
}
