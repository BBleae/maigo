package internal

import (
	"fmt"
	"log"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"time"

	"github.com/BurntSushi/toml"       // For TOML parsing (example)
	"github.com/Masterminds/semver/v3" // For semantic versioning (example)
	// You might need a timezone library if "dateutil.tz" functionality is strictly required beyond Go's standard library.
	// For basic timezone lookup, "time.LoadLocation" can be used.
)

// MaiVersion is the hardcoded version string, equivalent to Python's mai_version.
// You should define this constant appropriately.
const MaiVersion = "1.0.0-default" // Example value

// Version is an alias for semver.Version for convenience.
type Version = semver.Version

// SpecifierSet is an alias for semver.Constraints for convenience.
// It represents a set of version specifiers.
type SpecifierSet = semver.Constraints

// LLMConfigEntry represents the configuration for an LLM model.
// Corresponds to the structure of cfg_target in the Python model loading.
type LLMConfigEntry struct {
	Name     string   `toml:"name"`
	Provider string   `toml:"provider"` // e.g., "SILICONFLOW"
	BaseURL  string   `toml:"-"`        // Derived from provider, e.g., SILICONFLOW_BASE_URL
	Key      string   `toml:"-"`        // Derived from provider, e.g., SILICONFLOW_KEY
	Stream   bool     `toml:"stream"`
	PriIn    float64  `toml:"pri_in"`         // Price per 1k input tokens
	PriOut   float64  `toml:"pri_out"`        // Price per 1k output tokens
	Temp     *float64 `toml:"temp,omitempty"` // Pointer to allow omission / distinguish 0.0 from not set
	// Add other potential fields from TOML if necessary
}

// KeywordReactionRule defines the structure for a keyword reaction rule.
type KeywordReactionRule struct {
	Enable        bool     `toml:"enable"`
	RegexStrings  []string `toml:"regex"`
	ResponseTexts []string `toml:"response_texts"` // Example, adjust based on actual rule structure
	// Other fields like "action", "mood_change", etc.
	CompiledRegex []*regexp.Regexp `toml:"-"` // To be populated after loading
}

// BotConfig is the Go equivalent of the Python BotConfig dataclass.
type BotConfig struct {
	// Versioning
	InnerVersion *Version `toml:"-"` // Parsed from the "inner.version" field in TOML
	MaiVersion   string   `toml:"-"` // Hardcoded version

	// Bot settings
	BotQQ         string   `toml:"qq"` // Mapped from bot_config.get("qq")
	BotNickname   *string  `toml:"nickname,omitempty"`
	BotAliasNames []string `toml:"alias_names,omitempty"`

	// Group settings (from TOML lists, can be converted to maps for quick lookup if needed)
	TalkAllowedGroups       []string `toml:"talk_allowed,omitempty"`
	TalkFrequencyDownGroups []string `toml:"talk_frequency_down,omitempty"`
	BanUserID               []string `toml:"ban_user_id,omitempty"`
	TalkAllowedPrivate      []string `toml:"talk_allowed_private,omitempty"` // From experimental

	// Personality
	PersonalityCore  string   `toml:"personality_core"`
	PersonalitySides []string `toml:"personality_sides,omitempty"`

	// Identity
	IdentityDetail []string `toml:"identity_detail,omitempty"`
	Height         int      `toml:"height"` // cm
	Weight         int      `toml:"weight"` // kg
	Age            int      `toml:"age"`    // years
	Gender         string   `toml:"gender"`
	Appearance     string   `toml:"appearance"`

	// Schedule
	EnableScheduleGen           bool    `toml:"enable_schedule_gen"`
	PromptScheduleGen           string  `toml:"prompt_schedule_gen"`
	ScheduleDoingUpdateInterval int     `toml:"schedule_doing_update_interval"` // seconds
	ScheduleTemperature         float64 `toml:"schedule_temperature"`
	TimeZone                    string  `toml:"time_zone"`

	// Chat
	AllowFocusMode         bool             `toml:"allow_focus_mode"`
	BaseNormalChatNum      int              `toml:"base_normal_chat_num"`
	BaseFocusedChatNum     int              `toml:"base_focused_chat_num"`
	ObservationContextSize int              `toml:"observation_context_size"`
	MessageBuffer          bool             `toml:"message_buffer"`
	BanWords               []string         `toml:"ban_words,omitempty"`
	BanMsgsRegexStrings    []string         `toml:"ban_msgs_regex,omitempty"` // Raw regex strings
	CompiledBanMsgsRegex   []*regexp.Regexp `toml:"-"`                        // Compiled regexes

	// Focus Chat
	ReplyTriggerThreshold       float64 `toml:"reply_trigger_threshold"`
	DefaultDecayRatePerSecond   float64 `toml:"default_decay_rate_per_second"`
	ConsecutiveNoReplyThreshold int     `toml:"consecutive_no_reply_threshold"`
	CompressedLength            int     `toml:"compressed_length"`
	CompressLengthLimit         int     `toml:"compress_length_limit"`

	// Normal Chat
	ModelReasoningProbability       float64 `toml:"model_reasoning_probability"`
	ModelNormalProbability          float64 `toml:"model_normal_probability"`
	EmojiChance                     float64 `toml:"emoji_chance"`
	ThinkingTimeout                 int     `toml:"thinking_timeout"` // seconds
	WillingMode                     string  `toml:"willing_mode"`
	ResponseWillingAmplifier        float64 `toml:"response_willing_amplifier"`
	ResponseInterestedRateAmplifier float64 `toml:"response_interested_rate_amplifier"`
	DownFrequencyRate               float64 `toml:"down_frequency_rate"`
	EmojiResponsePenalty            float64 `toml:"emoji_response_penalty"`
	MentionedBotInevitableReply     bool    `toml:"mentioned_bot_inevitable_reply"`
	AtBotInevitableReply            bool    `toml:"at_bot_inevitable_reply"`

	// Emoji
	MaxEmojiNum        int    `toml:"max_emoji_num"`
	MaxReachDeletion   bool   `toml:"max_reach_deletion"`
	EmojiCheckInterval int    `toml:"check_interval"` // minutes
	SavePic            bool   `toml:"save_pic"`
	SaveEmoji          bool   `toml:"save_emoji"`
	StealEmoji         bool   `toml:"steal_emoji"`
	EmojiCheck         bool   `toml:"enable_check"`
	EmojiCheckPrompt   string `toml:"check_prompt"`

	// Memory
	BuildMemoryInterval              int       `toml:"build_memory_interval"` // seconds
	MemoryBuildDistribution          []float64 `toml:"memory_build_distribution,omitempty"`
	BuildMemorySampleNum             int       `toml:"build_memory_sample_num"`
	BuildMemorySampleLength          int       `toml:"build_memory_sample_length"`
	MemoryCompressRate               float64   `toml:"memory_compress_rate"`
	ForgetMemoryInterval             int       `toml:"forget_memory_interval"` // seconds
	MemoryForgetTime                 int       `toml:"memory_forget_time"`     // hours
	MemoryForgetPercentage           float64   `toml:"memory_forget_percentage"`
	ConsolidateMemoryInterval        int       `toml:"consolidate_memory_interval"` // seconds
	ConsolidationSimilarityThreshold float64   `toml:"consolidation_similarity_threshold"`
	ConsolidateMemoryPercentage      float64   `toml:"consolidate_memory_percentage"`
	MemoryBanWords                   []string  `toml:"memory_ban_words,omitempty"`

	// Mood
	MoodUpdateInterval  float64 `toml:"mood_update_interval"` // seconds
	MoodDecayRate       float64 `toml:"mood_decay_rate"`
	MoodIntensityFactor float64 `toml:"mood_intensity_factor"`

	// Keywords Reaction
	KeywordsReactionConfig struct {
		Enable bool                  `toml:"enable"`
		Rules  []KeywordReactionRule `toml:"rules,omitempty"`
	} `toml:"keywords_reaction"`

	// Chinese Typo
	ChineseTypoEnable          bool    `toml:"enable"`
	ChineseTypoErrorRate       float64 `toml:"error_rate"`
	ChineseTypoMinFreq         int     `toml:"min_freq"`
	ChineseTypoToneErrorRate   float64 `toml:"tone_error_rate"`
	ChineseTypoWordReplaceRate float64 `toml:"word_replace_rate"`

	// Response Splitter
	EnableKaomojiProtection bool `toml:"enable_kaomoji_protection"`
	EnableResponseSplitter  bool `toml:"enable_response_splitter"`
	ResponseMaxLength       int  `toml:"response_max_length"`
	ResponseMaxSentenceNum  int  `toml:"response_max_sentence_num"`
	ModelMaxOutputLength    int  `toml:"model_max_output_length"`

	// Remote
	RemoteEnable bool `toml:"enable"` // From remote.enable

	// Experimental
	EnableFriendChat  bool `toml:"enable_friend_chat"`
	EnablePFCChatting bool `toml:"pfc_chatting"` // Renamed from enable_pfc_chatting for TOML key

	// Model Config (LLM, VLM, Embedding, Moderation)
	// These will be pointers to allow them to be nil if not present in config
	LLMReasoning        *LLMConfigEntry `toml:"llm_reasoning,omitempty"`
	LLMNormal           *LLMConfigEntry `toml:"llm_normal,omitempty"`
	LLMTopicJudge       *LLMConfigEntry `toml:"llm_topic_judge,omitempty"`
	LLMSummary          *LLMConfigEntry `toml:"llm_summary,omitempty"`
	Embedding           *LLMConfigEntry `toml:"embedding,omitempty"`
	VLM                 *LLMConfigEntry `toml:"vlm,omitempty"`
	Moderation          *LLMConfigEntry `toml:"moderation,omitempty"`
	LLMObservation      *LLMConfigEntry `toml:"llm_observation,omitempty"`
	LLMSubHeartflow     *LLMConfigEntry `toml:"llm_sub_heartflow,omitempty"`
	LLMHeartflow        *LLMConfigEntry `toml:"llm_heartflow,omitempty"`
	LLMToolUse          *LLMConfigEntry `toml:"llm_tool_use,omitempty"`
	LLMPlan             *LLMConfigEntry `toml:"llm_plan,omitempty"`
	LLMPFCActionPlanner *LLMConfigEntry `toml:"llm_PFC_action_planner,omitempty"` // Added based on Python list
	LLMPFCChat          *LLMConfigEntry `toml:"llm_PFC_chat,omitempty"`           // Added based on Python list
	LLMPFCReplyChecker  *LLMConfigEntry `toml:"llm_PFC_reply_checker,omitempty"`  // Added based on Python list

	// API URLs for platforms
	APIURLs map[string]string `toml:"platforms,omitempty"` // Stores platform URLs, key is platform name

	// Raw Toml Data for dynamic access if needed, or for sections not fully mapped
	rawTomlData map[string]interface{} `toml:"-"`
}

// NewBotConfig creates a new BotConfig with default values.
func NewBotConfig() *BotConfig {
	// Default values from the Python dataclass
	return &BotConfig{
		MaiVersion: MaiVersion, // Hardcoded version

		// Bot
		BotQQ:         "114514",
		BotAliasNames: []string{},

		// Group (initialized as empty slices, will be populated from TOML)
		TalkAllowedGroups:       []string{},
		TalkFrequencyDownGroups: []string{},
		BanUserID:               []string{},
		TalkAllowedPrivate:      []string{},

		// Personality
		PersonalityCore: "用一句话或几句话描述人格的核心特点", // "Describe core personality in one or a few sentences"
		PersonalitySides: []string{
			"用一句话或几句话描述人格的一些侧面", // "Describe some aspects of personality"
			"用一句话或几句话描述人格的一些侧面",
			"用一句话或几句话描述人格的一些侧面",
		},
		// Identity
		IdentityDetail: []string{
			"身份特点", // "Identity characteristic"
			"身份特点",
		},
		Height:     170,
		Weight:     50,
		Age:        20,
		Gender:     "男",          // "Male"
		Appearance: "用几句话描述外貌特征", // "Describe appearance in a few sentences"

		// Schedule
		EnableScheduleGen:           false,
		PromptScheduleGen:           "无日程", // "No schedule"
		ScheduleDoingUpdateInterval: 300,
		ScheduleTemperature:         0.5,
		TimeZone:                    "Asia/Shanghai",

		// Chat
		AllowFocusMode:         true,
		BaseNormalChatNum:      3,
		BaseFocusedChatNum:     2,
		ObservationContextSize: 12,
		MessageBuffer:          true,
		BanWords:               []string{},
		BanMsgsRegexStrings:    []string{},

		// Focus Chat
		ReplyTriggerThreshold:       3.0,
		DefaultDecayRatePerSecond:   0.98,
		ConsecutiveNoReplyThreshold: 3,
		CompressedLength:            5,
		CompressLengthLimit:         5,

		// Normal Chat
		ModelReasoningProbability:       0.7,
		ModelNormalProbability:          0.3,
		EmojiChance:                     0.2,
		ThinkingTimeout:                 120,
		WillingMode:                     "classical",
		ResponseWillingAmplifier:        1.0,
		ResponseInterestedRateAmplifier: 1.0,
		DownFrequencyRate:               3.0,
		EmojiResponsePenalty:            0.0,
		MentionedBotInevitableReply:     false,
		AtBotInevitableReply:            false,

		// Emoji
		MaxEmojiNum:        200,
		MaxReachDeletion:   true,
		EmojiCheckInterval: 120, // minutes
		SavePic:            false,
		SaveEmoji:          false,
		StealEmoji:         true,
		EmojiCheck:         false,
		EmojiCheckPrompt:   "符合公序良俗", // "Complies with public order and good customs"

		// Memory
		BuildMemoryInterval:              600,
		MemoryBuildDistribution:          []float64{4, 2, 0.6, 24, 8, 0.4},
		BuildMemorySampleNum:             10,
		BuildMemorySampleLength:          20,
		MemoryCompressRate:               0.1,
		ForgetMemoryInterval:             600,
		MemoryForgetTime:                 24, // hours
		MemoryForgetPercentage:           0.01,
		ConsolidateMemoryInterval:        1000,
		ConsolidationSimilarityThreshold: 0.7,
		ConsolidateMemoryPercentage:      0.01,
		MemoryBanWords:                   []string{"表情包", "图片", "回复", "聊天记录"}, // "emoji", "image", "reply", "chat history"

		// Mood
		MoodUpdateInterval:  1.0,
		MoodDecayRate:       0.95,
		MoodIntensityFactor: 0.7,

		// Keywords Reaction
		KeywordsReactionConfig: struct {
			Enable bool                  `toml:"enable"`
			Rules  []KeywordReactionRule `toml:"rules,omitempty"`
		}{Enable: false, Rules: []KeywordReactionRule{}},

		// Chinese Typo
		ChineseTypoEnable:          true,
		ChineseTypoErrorRate:       0.03,
		ChineseTypoMinFreq:         7,
		ChineseTypoToneErrorRate:   0.2,
		ChineseTypoWordReplaceRate: 0.02,

		// Response Splitter
		EnableKaomojiProtection: false,
		EnableResponseSplitter:  true,
		ResponseMaxLength:       100,
		ResponseMaxSentenceNum:  3,
		ModelMaxOutputLength:    800,

		// Remote
		RemoteEnable: true,

		// Experimental
		EnableFriendChat:  false,
		EnablePFCChatting: false,

		// Model Configs (initialized to nil, to be populated from TOML)
		LLMReasoning:        nil,
		LLMNormal:           nil,
		LLMTopicJudge:       nil,
		LLMSummary:          nil,
		Embedding:           nil,
		VLM:                 nil,
		Moderation:          nil,
		LLMObservation:      nil,
		LLMSubHeartflow:     nil,
		LLMHeartflow:        nil,
		LLMToolUse:          nil,
		LLMPlan:             nil,
		LLMPFCActionPlanner: nil,
		LLMPFCChat:          nil,
		LLMPFCReplyChecker:  nil,

		APIURLs: make(map[string]string),
	}
}

// GetConfigDir retrieves the configuration directory.
// Equivalent to the Python static method get_config_dir.
func GetConfigDir() (string, error) {
	//exePath, err := os.Executable()
	//if err != nil {
	//	return "", fmt.Errorf("failed to get executable path: %w", err)
	//}
	//currentDir := filepath.Dir(exePath) // Or use "." for current working directory if more appropriate
	// The Python code uses os.path.abspath(__file__), then goes up two directories.
	// This needs to be adjusted based on your Go project structure.
	// Assuming the config dir is relative to the executable or a known path.
	// For this example, let's assume it's a 'config' subdir in the parent of currentDir.
	// This logic might need to be adapted. If __file__ implies the source file location,
	// runtime determination in Go is different. Often, config paths are passed as flags
	// or environment variables, or are relative to the working directory.

	// Simplified: assume config dir is 'config' in the project root.
	// This is a common Go pattern if you run from project root.
	// Or, if your binary is in cmd/myapp/, and config is in root/config:
	// rootDir := filepath.Join(currentDir, "..", "..") // If currentDir is deep
	// For a simpler example, let's assume config is in ./config relative to where binary is run
	// or a path determined by build tags/environment.

	// Python: os.path.abspath(os.path.join(current_dir, "..", "..", "config"))
	// Let's assume a structure where the binary is in a 'bin' folder and config is at project_root/config
	// If currentDir is where the binary is:
	// projectRootDir := filepath.Dir(currentDir) // if binary is in project_root/bin/
	// configDir := filepath.Join(projectRootDir, "config")

	// A more robust way for Go is often to make the config path configurable.
	// For this translation, I'll try to mimic the Python logic assuming a certain structure.
	// If `current_dir` in Python refers to the directory of the .py file:
	// Go's `os.Executable()` gives the path to the compiled binary.
	// If the Python script is `myproject/internal/config/config.py`, then `current_dir` is `myproject/internal/config`.
	// `root_dir` becomes `myproject`. `config_dir` becomes `myproject/config`.

	// Let's assume the Go executable is at `myproject/bin/app` and config is `myproject/config`.
	// Then `currentDir` = `myproject/bin`. `rootDir` = `myproject`.
	// rootDir := filepath.Dir(currentDir)
	// configDir := filepath.Join(rootDir, "config")

	// Fallback to a simpler assumption: config directory is named "config"
	// in the current working directory or a path provided.
	// For direct translation of the Python logic, it's tricky without knowing the execution context.
	// Let's use a common Go practice: config directory relative to working directory.
	configDir := "config" // Or use an absolute path or env var

	if _, err := os.Stat(configDir); os.IsNotExist(err) {
		if mkErr := os.MkdirAll(configDir, 0755); mkErr != nil {
			return "", fmt.Errorf("failed to create config directory '%s': %w", configDir, mkErr)
		}
	}
	absConfigDir, err := filepath.Abs(configDir)
	if err != nil {
		return "", fmt.Errorf("failed to get absolute path for config directory '%s': %w", configDir, err)
	}
	return absConfigDir, nil
}

// ConvertToSpecifierSet converts a version constraint string to a SpecifierSet.
// Equivalent to the Python class method convert_to_specifierset.
func ConvertToSpecifierSet(value string) (*SpecifierSet, error) {
	constraints, err := semver.NewConstraint(value)
	if err != nil {
		log.Printf("Error: '%s' uses an invalid version constraint expression: %v. Please read https://semver.org/", value, err)
		return nil, fmt.Errorf("invalid version constraint '%s': %w", value, err)
	}
	return constraints, nil
}

// GetConfigVersion extracts and parses the version from the TOML data.
// Equivalent to the Python class method get_config_version.
func GetConfigVersion(tomlData map[string]interface{}) (*Version, error) {
	var configVersionStr string
	innerSection, ok := tomlData["inner"].(map[string]interface{})
	if !ok {
		// Python code adds inner if not present, and defaults version to "0.0.0"
		// This behavior might be desired or an error in Go.
		// For now, let's assume it's an error if 'inner' or 'inner.version' is missing for clarity.
		// Or, replicate Python:
		log.Println("Warning: 'inner' section not found in TOML, defaulting version to 0.0.0")
		configVersionStr = "0.0.0"
	} else {
		versionVal, found := innerSection["version"]
		if !found {
			log.Println("Warning: 'inner.version' key not found in TOML, defaulting version to 0.0.0")
			configVersionStr = "0.0.0" // Default if key is missing
		} else {
			configVersionStr, ok = versionVal.(string)
			if !ok {
				return nil, fmt.Errorf("'inner.version' is not a string")
			}
		}
	}

	ver, err := semver.NewVersion(configVersionStr)
	if err != nil {
		log.Printf("Error: 'inner.version' ('%s') is an invalid version string: %v. Please check https://semver.org/", configVersionStr, err)
		return nil, fmt.Errorf("invalid version string in 'inner.version': %w", err)
	}
	return ver, nil
}

// LoadConfig loads the BotConfig from a TOML file.
// This is a complex function. The implementation below is a skeleton
// and would need to be fleshed out considerably to match all the Python logic,
// especially the version-based conditional loading and error handling.
func LoadConfig(configPath string) (*BotConfig, error) {
	cfg := NewBotConfig() // Start with defaults

	if configPath == "" {
		// Default config path logic if desired, e.g., from GetConfigDir()
		defaultDir, err := GetConfigDir()
		if err != nil {
			return nil, fmt.Errorf("could not determine default config directory: %w", err)
		}
		configPath = filepath.Join(defaultDir, "bot_config.toml") // Default filename
	}

	log.Printf("Attempting to load configuration from: %s", configPath)

	if _, err := os.Stat(configPath); os.IsNotExist(err) {
		log.Printf("Config file '%s' not found. Using default configuration.", configPath)
		// Potentially save a default config file here if that's desired behavior
		// For now, just return the default config.
		// Before returning, compile any default regexes if necessary
		if err := compileRegexes(cfg); err != nil {
			return nil, fmt.Errorf("error compiling default regexes: %w", err)
		}
		return cfg, nil // Return default config if file doesn't exist
	}

	// Read the TOML file
	tomlData, err := os.ReadFile(configPath)
	if err != nil {
		return nil, fmt.Errorf("failed to read config file '%s': %w", configPath, err)
	}

	// First, unmarshal into a generic map to extract version and handle dynamic parts
	var rawTomlMap map[string]interface{}
	if err := toml.Unmarshal(tomlData, &rawTomlMap); err != nil {
		// Try to provide line and column for TOMLDecodeError like Python
		// Note: BurntSushi/toml.Decode returns TomlDecodeError which has line number info.
		// We'd need to use toml.Decode() instead of toml.Unmarshal() for that.
		return nil, fmt.Errorf("failed to parse TOML from '%s': %w. Check TOML syntax.", configPath, err)
	}
	cfg.rawTomlData = rawTomlMap

	// Get config file version
	innerVersion, err := GetConfigVersion(rawTomlMap)
	if err != nil {
		return nil, fmt.Errorf("failed to get config version from '%s': %w", configPath, err)
	}
	cfg.InnerVersion = innerVersion
	log.Printf("Successfully parsed config version: %s", cfg.InnerVersion.String())

	// Now, unmarshal into the struct. Some fields might be overridden if present in TOML.
	// This will overwrite defaults set by NewBotConfig() with values from the TOML file.
	if err := toml.Unmarshal(tomlData, cfg); err != nil {
		return nil, fmt.Errorf("failed to unmarshal TOML into BotConfig struct: %w", err)
	}

	// The Python code has a sophisticated `include_configs` map for version-dependent loading.
	// Replicating that dispatch table and conditional logic for each section:
	// For each section (bot, groups, personality, model, etc.):
	// 1. Check if the section exists in rawTomlMap.
	// 2. Get the 'support' version specifier string for that section.
	// 3. Convert it to a SpecifierSet (semver.Constraints).
	// 4. Check if cfg.InnerVersion satisfies this SpecifierSet.
	// 5. If supported, apply the values from that section of rawTomlMap to cfg.
	//    This might involve custom logic for each section, similar to the Python
	//    nested functions (personality(parent), identity(parent), etc.).
	//    The direct unmarshal above handles many cases, but version checks and
	//    more complex assignments (like for LLMConfigEntry construction) need specific handling.

	// Example for handling the 'model' section's dynamic LLM entries:
	if modelSectionRaw, ok := rawTomlMap["model"].(map[string]interface{}); ok {
		llmConfigList := []string{
			"llm_reasoning", "llm_normal", "llm_topic_judge", "llm_summary", "vlm",
			"embedding", "llm_tool_use", "llm_observation", "llm_sub_heartflow",
			"llm_plan", "llm_heartflow", "llm_PFC_action_planner", "llm_PFC_chat", "llm_PFC_reply_checker",
			"moderation", // Ensure moderation is also in this list if handled similarly
		}

		// Required version for this model loading logic (example)
		modelLogicConstraint, _ := ConvertToSpecifierSet(">=0.0.1")

		for _, itemName := range llmConfigList {
			if itemDataRaw, ok := modelSectionRaw[itemName].(map[string]interface{}); ok {
				entry := &LLMConfigEntry{}
				// Manually map fields from itemDataRaw to entry, checking types
				if name, ok := itemDataRaw["name"].(string); ok {
					entry.Name = name
				}
				if provider, ok := itemDataRaw["provider"].(string); ok {
					entry.Provider = provider
				}
				// Stream, PriIn, PriOut, Temp need similar type assertions and assignments
				if stream, ok := itemDataRaw["stream"].(bool); ok {
					entry.Stream = stream
				}
				if priIn, ok := itemDataRaw["pri_in"].(float64); ok {
					entry.PriIn = priIn
				} // TOML numbers are float64
				if priOut, ok := itemDataRaw["pri_out"].(float64); ok {
					entry.PriOut = priOut
				}
				if temp, ok := itemDataRaw["temp"].(float64); ok {
					entry.Temp = &temp
				}

				// Python logic for base_url and key based on provider (e.g., "OPENAI_BASE_URL", "OPENAI_KEY")
				// This often involves looking up environment variables in Go.
				if entry.Provider != "" {
					entry.BaseURL = os.Getenv(strings.ToUpper(entry.Provider) + "_BASE_URL")
					entry.Key = os.Getenv(strings.ToUpper(entry.Provider) + "_KEY")
				}

				// Assign to the correct field in cfg using a switch or reflection (reflection is slower and more complex)
				// This part is simplified; direct assignment is better if possible.
				// The Python version uses setattr. In Go, you'd typically use a switch or if-else chain.
				switch itemName {
				case "llm_reasoning":
					cfg.LLMReasoning = entry
				case "llm_normal":
					cfg.LLMNormal = entry
				// ... other cases for all llm items
				case "llm_topic_judge":
					cfg.LLMTopicJudge = entry
				case "llm_summary":
					cfg.LLMSummary = entry
				case "embedding":
					cfg.Embedding = entry
				case "vlm":
					cfg.VLM = entry
				case "moderation":
					cfg.Moderation = entry
				case "llm_observation":
					cfg.LLMObservation = entry
				case "llm_sub_heartflow":
					cfg.LLMSubHeartflow = entry
				case "llm_heartflow":
					cfg.LLMHeartflow = entry
				case "llm_tool_use":
					cfg.LLMToolUse = entry
				case "llm_plan":
					cfg.LLMPlan = entry
				case "llm_PFC_action_planner":
					cfg.LLMPFCActionPlanner = entry
				case "llm_PFC_chat":
					cfg.LLMPFCChat = entry
				case "llm_PFC_reply_checker":
					cfg.LLMPFCReplyChecker = entry
				default:
					log.Printf("Warning: Unhandled LLM config item in LoadConfig: %s", itemName)
				}

				// Version check for specific fields within this LLM entry
				if cfg.InnerVersion != nil && modelLogicConstraint != nil {
					if !modelLogicConstraint.Check(cfg.InnerVersion) {
						// Handle incompatible version for this specific model structure if needed
						log.Printf("Warning: Model config for '%s' might be outdated for config version %s", itemName, cfg.InnerVersion.String())
					}
				}
			} else if itemDataRaw != nil {
				log.Printf("Warning: Model item '%s' is not a map (dictionary) in TOML, skipping.", itemName)
			}
			// If item not in model_config, Python raises KeyError. Here, it would just be nil.
		}
	}

	// Example for schedule.time_zone validation
	if cfg.TimeZone != "" {
		_, err := time.LoadLocation(cfg.TimeZone)
		if err != nil {
			log.Printf("Warning: Invalid timezone '%s', using default '%s'. Error: %v", cfg.TimeZone, NewBotConfig().TimeZone, err)
			cfg.TimeZone = NewBotConfig().TimeZone // Fallback to default
		}
	}

	// Post-processing: Compile regexes
	if err := compileRegexes(cfg); err != nil {
		return nil, fmt.Errorf("error compiling regexes from config: %w", err)
	}

	// Final checks (e.g., identity_detail not empty)
	if len(cfg.IdentityDetail) == 0 && cfg.InnerVersion != nil /* and version requires it */ {
		// The Python code raises error if identity_detail is empty.
		// Depending on version constraints, this check might be conditional.
		// Example: identityConstraint, _ := ConvertToSpecifierSet(">=1.2.4")
		// if identityConstraint.Check(cfg.InnerVersion) && len(cfg.IdentityDetail) == 0 { ... }
		log.Println("Warning: 'identity_detail' is empty. This might be an error depending on config version requirements.")
		// return nil, fmt.Errorf("config error: [identity]identity_detail cannot be empty for version %s", cfg.InnerVersion.String())
	}

	log.Printf("Successfully loaded and processed configuration from: %s", configPath)
	return cfg, nil
}

// compileRegexes compiles regex strings in the config into regexp.Regexp objects.
func compileRegexes(cfg *BotConfig) error {
	// Compile BanMsgsRegex
	cfg.CompiledBanMsgsRegex = make([]*regexp.Regexp, 0, len(cfg.BanMsgsRegexStrings))
	for _, s := range cfg.BanMsgsRegexStrings {
		re, err := regexp.Compile(s)
		if err != nil {
			return fmt.Errorf("failed to compile ban_msgs_regex '%s': %w", s, err)
		}
		cfg.CompiledBanMsgsRegex = append(cfg.CompiledBanMsgsRegex, re)
	}

	// Compile regexes in KeywordReactionRules
	if cfg.KeywordsReactionConfig.Enable {
		for i := range cfg.KeywordsReactionConfig.Rules {
			rule := &cfg.KeywordsReactionConfig.Rules[i] // Get pointer to modify
			if rule.Enable && len(rule.RegexStrings) > 0 {
				rule.CompiledRegex = make([]*regexp.Regexp, 0, len(rule.RegexStrings))
				for _, s := range rule.RegexStrings {
					re, err := regexp.Compile(s)
					if err != nil {
						// Log error and skip this regex, or return error to fail loading
						log.Printf("Error compiling keyword_reaction regex '%s': %v. Skipping this regex.", s, err)
						// return fmt.Errorf("failed to compile keyword_reaction regex '%s': %w", s, err)
					} else {
						rule.CompiledRegex = append(rule.CompiledRegex, re)
					}
				}
			}
		}
	}
	return nil
}

// Helper function to get a value from a map[string]interface{} with a default.
// Not strictly necessary if using TOML unmarshal directly into struct with defaults,
// but useful if manually parsing sections like the Python code does.
func getFromMap[T any](m map[string]interface{}, key string, defaultValue T) T {
	if val, ok := m[key]; ok {
		if typedVal, typeOK := val.(T); typeOK {
			return typedVal
		}
	}
	return defaultValue
}

func main() {
	// Example Usage:
	// 1. Get config directory
	configDir, err := GetConfigDir()
	if err != nil {
		log.Fatalf("Error getting config directory: %v", err)
	}
	log.Printf("Config directory: %s", configDir)

	// 2. Define path to config file
	configFile := filepath.Join(configDir, "bot_config.toml")
	// Ensure a dummy bot_config.toml exists in ./config/ for this example to run.
	// Example bot_config.toml:
	/*
		[inner]
		version = "1.6.0"

		[bot]
		qq = "123456789"
		nickname = "GoBot"
		alias_names = ["BotGo", "Go机器人"]

		[personality]
		personality_core = "A helpful Go assistant."
		# ... other sections
	*/

	// Create a dummy config file for testing if it doesn't exist
	if _, err := os.Stat(configFile); os.IsNotExist(err) {
		log.Printf("Creating dummy config file at %s", configFile)
		dummyToml := `
[inner]
version = "1.6.0" # Make sure this version is supported by your logic

[bot]
qq = "987654321"
nickname = "GoTestBot"
alias_names = ["Tester"]

[personality]
personality_core = "A test personality."
personality_sides = ["Side A", "Side B"]

[identity]
identity_detail = ["Test Detail 1", "Test Detail 2"]
height = 160
weight = 55
age = 2
gender = "未知"
appearance = "Shiny metal"

[schedule]
enable_schedule_gen = true
prompt_schedule_gen = "Current task: testing."
time_zone = "America/New_York"

[chat] # Assuming chat section is supported by version 1.6.0
allow_focus_mode = true
ban_words = ["badword1", "badword2"]
ban_msgs_regex = ["^spammy.*", ".*phishing.*"]

[model.llm_reasoning]
name = "gpt-4"
provider = "OPENAI" # Needs OPENAI_BASE_URL and OPENAI_KEY env vars
stream = true
pri_in = 0.03
pri_out = 0.06
temp = 0.7

[keywords_reaction]
enable = true
rules = [
  { enable = true, regex = ["hello bot"], response_texts = ["Hello there!"] },
  { enable = true, regex = ["help me"], response_texts = ["How can I assist?"] }
]
`
		err := os.MkdirAll(filepath.Dir(configFile), 0755)
		if err != nil {
			return
		}
		if err := os.WriteFile(configFile, []byte(dummyToml), 0644); err != nil {
			log.Fatalf("Failed to write dummy config: %v", err)
		}
		// Set dummy env vars for LLM config
		err = os.Setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
		if err != nil {
			return
		}
		err = os.Setenv("OPENAI_KEY", "sk-dummykey")
		if err != nil {
			return
		}
	}

	// 3. Load config
	cfg, err := LoadConfig(configFile)
	if err != nil {
		log.Fatalf("Error loading config: %v", err)
	}

	log.Printf("Loaded Bot QQ: %s", cfg.BotQQ)
	log.Printf("Loaded Bot Nickname: %v", cfg.BotNickname) // Dereference if not nil
	if cfg.InnerVersion != nil {
		log.Printf("Config Inner Version: %s", cfg.InnerVersion.String())
	}
	if cfg.LLMReasoning != nil {
		log.Printf("LLM Reasoning Name: %s, Provider: %s, BaseURL: %s", cfg.LLMReasoning.Name, cfg.LLMReasoning.Provider, cfg.LLMReasoning.BaseURL)
	}
	if len(cfg.CompiledBanMsgsRegex) > 0 {
		log.Printf("Compiled ban message regex: %s", cfg.CompiledBanMsgsRegex[0].String())
	}
	if cfg.KeywordsReactionConfig.Enable && len(cfg.KeywordsReactionConfig.Rules) > 0 && len(cfg.KeywordsReactionConfig.Rules[0].CompiledRegex) > 0 {
		log.Printf("First keyword reaction rule, first compiled regex: %s", cfg.KeywordsReactionConfig.Rules[0].CompiledRegex[0].String())
	}

}
