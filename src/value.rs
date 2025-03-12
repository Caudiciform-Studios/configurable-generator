use std::fmt;

use lazy_regex::regex_switch;
use rand::prelude::*;
use serde::{de, Deserialize, Deserializer, Serialize, Serializer};

use crate::Ctx;

#[derive(Copy, Clone, Debug)]
pub enum Value {
    Const(f64),
    Range(f64, f64),
    DynamicRange(f64, f64),
}

impl Default for Value {
    fn default() -> Value {
        Value::Const(0.0)
    }
}

impl Value {
    pub fn solidify(&mut self, ctx: &mut Ctx) {
        if let Value::Range(start, end) = self {
            *self = Value::Const(ctx.rng.gen_range(*start..*end));
        }
    }

    pub fn val(&self, rng: &mut impl Rng) -> f64 {
        match self {
            Value::Const(v) => *v,
            Value::Range(start, _end) => *start,
            Value::DynamicRange(start, end) => rng.gen_range(*start..*end),
        }
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Value::Const(v) => write!(f, "{v}"),
            Value::Range(s, e) => write!(f, "{s}..{e}"),
            Value::DynamicRange(s, e) => write!(f, "d{s}..{e}"),
        }
    }
}

impl std::str::FromStr for Value {
    type Err = &'static str;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        regex_switch!(
            s,
            "d(?<s>-?[0-9]+(:?\\.[0-9]+)?)\\.\\.(?<e>-?[0-9]+(:?\\.[0-9]+)?)" => Value::DynamicRange(s.parse().unwrap(), e.parse().unwrap()),
            "(?<s>-?[0-9]+(:?\\.[0-9]+)?)\\.\\.(?<e>-?[0-9]+(:?\\.[0-9]+)?)" => Value::Range(s.parse().unwrap(), e.parse().unwrap()),
            "(?<n>-?[0-9]+(:?\\.[0-9]+)?)" => Value::Const(n.parse().unwrap()),
        ).ok_or("Fail")
    }
}

impl Serialize for Value {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.collect_str(self)
    }
}
impl<'de> Deserialize<'de> for Value {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        <Self as std::str::FromStr>::from_str(&s).map_err(de::Error::custom)
    }
}
