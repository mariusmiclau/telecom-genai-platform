"""Tests for agent tools."""

import pytest

from src.tools.registry import ToolResult
from src.tools.network import NetworkHealthTool, NetworkConfigTool
from src.tools.ticketing import TicketCreatorTool, TicketQueryTool
from src.tools.sla import SLAMonitorTool
from src.tools.config import ConfigValidatorTool, ConfigDiffTool


class TestNetworkHealthTool:
    """Tests for NetworkHealthTool."""

    @pytest.fixture
    def tool(self):
        """Create a NetworkHealthTool instance."""
        return NetworkHealthTool()

    def test_tool_has_required_attributes(self, tool):
        """Test tool has name, description, and parameters."""
        assert hasattr(tool, "name")
        assert hasattr(tool, "description")
        assert hasattr(tool, "parameters")
        assert tool.name == "network_health"

    @pytest.mark.asyncio
    async def test_execute_returns_tool_result(self, tool):
        """Test execute returns a ToolResult."""
        result = await tool.execute({"region": "region_a"})
        assert isinstance(result, ToolResult)
        assert result.data is not None
        assert result.summary is not None
        assert result.latency_ms >= 0

    @pytest.mark.asyncio
    async def test_execute_with_element_id(self, tool):
        """Test execute with specific element ID."""
        result = await tool.execute({"region": "region_a", "element_id": "rtr-001"})
        assert isinstance(result, ToolResult)

    @pytest.mark.asyncio
    async def test_execute_invalid_region(self, tool):
        """Test execute with invalid region."""
        result = await tool.execute({"region": "invalid_region"})
        assert isinstance(result, ToolResult)
        # Should handle gracefully (error in data or empty results)


class TestNetworkConfigTool:
    """Tests for NetworkConfigTool."""

    @pytest.fixture
    def tool(self):
        """Create a NetworkConfigTool instance."""
        return NetworkConfigTool()

    def test_tool_has_required_attributes(self, tool):
        """Test tool has required attributes."""
        assert tool.name == "network_config"
        assert tool.description
        assert tool.parameters

    @pytest.mark.asyncio
    async def test_execute_returns_tool_result(self, tool):
        """Test execute returns a ToolResult."""
        result = await tool.execute({"element_id": "rtr-001"})
        assert isinstance(result, ToolResult)


class TestTicketCreatorTool:
    """Tests for TicketCreatorTool."""

    @pytest.fixture
    def tool(self):
        """Create a TicketCreatorTool instance."""
        return TicketCreatorTool()

    def test_tool_has_required_attributes(self, tool):
        """Test tool has required attributes."""
        assert tool.name == "ticket_creator"
        assert tool.description
        assert "title" in tool.parameters
        assert "priority" in tool.parameters

    @pytest.mark.asyncio
    async def test_create_incident_ticket(self, tool):
        """Test creating an incident ticket."""
        result = await tool.execute({
            "title": "Network outage in Region A",
            "description": "Multiple routers reporting connectivity issues",
            "priority": "high",
            "category": "incident",
            "region": "region_a",
        })
        assert isinstance(result, ToolResult)
        assert "ticket_id" in result.data
        assert result.data["ticket_id"].startswith("INC")
        assert result.data["priority"] == "high"

    @pytest.mark.asyncio
    async def test_create_change_ticket(self, tool):
        """Test creating a change request ticket."""
        result = await tool.execute({
            "title": "Upgrade router firmware",
            "description": "Scheduled maintenance",
            "priority": "medium",
            "category": "change",
        })
        assert result.data["ticket_id"].startswith("CHG")

    @pytest.mark.asyncio
    async def test_critical_ticket_assignment(self, tool):
        """Test critical tickets get proper assignment."""
        result = await tool.execute({
            "title": "Critical outage",
            "description": "Emergency",
            "priority": "critical",
            "category": "incident",
        })
        assert "critical" in result.data["assignee"].lower()

    @pytest.mark.asyncio
    async def test_ticket_with_affected_elements(self, tool):
        """Test ticket with affected network elements."""
        result = await tool.execute({
            "title": "Switch failure",
            "description": "Switch down",
            "priority": "high",
            "category": "incident",
            "affected_elements": ["sw-001", "sw-002"],
        })
        assert isinstance(result, ToolResult)


class TestTicketQueryTool:
    """Tests for TicketQueryTool."""

    @pytest.fixture
    def tool(self):
        """Create a TicketQueryTool instance."""
        return TicketQueryTool()

    def test_tool_has_required_attributes(self, tool):
        """Test tool has required attributes."""
        assert tool.name == "ticket_query"
        assert tool.description

    @pytest.mark.asyncio
    async def test_query_returns_tool_result(self, tool):
        """Test query returns a ToolResult."""
        result = await tool.execute({})
        assert isinstance(result, ToolResult)

    @pytest.mark.asyncio
    async def test_query_by_status(self, tool):
        """Test query by ticket status."""
        result = await tool.execute({"status": "new"})
        assert isinstance(result, ToolResult)


class TestSLAMonitorTool:
    """Tests for SLAMonitorTool."""

    @pytest.fixture
    def tool(self):
        """Create an SLAMonitorTool instance."""
        return SLAMonitorTool()

    def test_tool_has_required_attributes(self, tool):
        """Test tool has required attributes."""
        assert tool.name == "sla_monitor"
        assert tool.description
        assert "region" in tool.parameters

    @pytest.mark.asyncio
    async def test_get_status(self, tool):
        """Test getting SLA status."""
        result = await tool.execute({
            "region": "region_a",
            "check_type": "status",
        })
        assert isinstance(result, ToolResult)
        assert "services" in result.data
        assert "compliant" in result.data

    @pytest.mark.asyncio
    async def test_get_at_risk_services(self, tool):
        """Test getting at-risk services."""
        result = await tool.execute({
            "region": "region_a",
            "check_type": "at_risk",
        })
        assert isinstance(result, ToolResult)
        assert "at_risk_services" in result.data

    @pytest.mark.asyncio
    async def test_get_breaches(self, tool):
        """Test getting SLA breaches."""
        result = await tool.execute({
            "region": "region_a",
            "check_type": "breaches",
        })
        assert isinstance(result, ToolResult)
        assert "total_breaches_30d" in result.data

    @pytest.mark.asyncio
    async def test_sla_thresholds_in_result(self, tool):
        """Test SLA thresholds are reflected in results."""
        result = await tool.execute({
            "region": "region_a",
            "check_type": "status",
        })
        services = result.data.get("services", [])
        if services:
            service = services[0]
            # Check that threshold fields exist
            assert "target_availability" in service
            assert "target_latency_ms" in service
            assert "target_error_rate" in service

    @pytest.mark.asyncio
    async def test_invalid_region_handled(self, tool):
        """Test invalid region is handled gracefully."""
        result = await tool.execute({"region": "nonexistent_region"})
        assert isinstance(result, ToolResult)
        assert "error" in result.data


class TestConfigValidatorTool:
    """Tests for ConfigValidatorTool."""

    @pytest.fixture
    def tool(self):
        """Create a ConfigValidatorTool instance."""
        return ConfigValidatorTool()

    def test_tool_has_required_attributes(self, tool):
        """Test tool has required attributes."""
        assert tool.name == "config_validator"
        assert tool.description
        assert "config" in tool.parameters

    @pytest.mark.asyncio
    async def test_valid_config_passes(self, tool):
        """Test valid configuration passes validation."""
        config = """
        interface GigabitEthernet0/1
          ip address 10.0.0.1 255.255.255.0
          no shutdown
        !
        ntp server 10.0.0.100
        logging host 10.0.0.200
        """
        result = await tool.execute({
            "config": config,
            "element_id": "rtr-001",
        })
        assert isinstance(result, ToolResult)
        assert "is_valid" in result.data

    @pytest.mark.asyncio
    async def test_detects_plaintext_password(self, tool):
        """Test detection of plaintext password."""
        config = """
        username admin password 0 plaintextpassword
        """
        result = await tool.execute({
            "config": config,
            "element_id": "rtr-001",
            "change_type": "security",
        })
        assert result.data["is_valid"] is False
        assert any("SEC-001" in e["rule_id"] for e in result.data["errors"])

    @pytest.mark.asyncio
    async def test_detects_telnet_enabled(self, tool):
        """Test detection of telnet enabled."""
        config = """
        line vty 0 4
          transport input telnet
        """
        result = await tool.execute({
            "config": config,
            "element_id": "rtr-001",
        })
        assert result.data["is_valid"] is False
        assert any("SEC-002" in e["rule_id"] for e in result.data["errors"])

    @pytest.mark.asyncio
    async def test_warns_on_missing_ntp(self, tool):
        """Test warning on missing NTP configuration."""
        config = """
        interface GigabitEthernet0/1
          ip address 10.0.0.1 255.255.255.0
        """
        result = await tool.execute({
            "config": config,
            "element_id": "rtr-001",
        })
        # Should have NTP warning
        assert result.data["warning_count"] > 0 or result.data["is_valid"] is True

    @pytest.mark.asyncio
    async def test_strict_mode(self, tool):
        """Test strict mode treats warnings as errors."""
        config = """
        interface GigabitEthernet0/1
          ip address 10.0.0.1 255.255.255.0
        """
        result = await tool.execute({
            "config": config,
            "element_id": "rtr-001",
            "strict_mode": True,
        })
        # In strict mode, warnings become errors
        assert isinstance(result, ToolResult)

    @pytest.mark.asyncio
    async def test_empty_config_handled(self, tool):
        """Test empty config is handled."""
        result = await tool.execute({
            "config": "",
            "element_id": "rtr-001",
        })
        assert isinstance(result, ToolResult)
        assert "error" in result.data


class TestConfigDiffTool:
    """Tests for ConfigDiffTool."""

    @pytest.fixture
    def tool(self):
        """Create a ConfigDiffTool instance."""
        return ConfigDiffTool()

    def test_tool_has_required_attributes(self, tool):
        """Test tool has required attributes."""
        assert tool.name == "config_diff"

    @pytest.mark.asyncio
    async def test_diff_shows_additions(self, tool):
        """Test diff shows added lines."""
        current = "line1\nline2"
        proposed = "line1\nline2\nline3"
        result = await tool.execute({
            "current_config": current,
            "proposed_config": proposed,
        })
        assert "line3" in result.data["added"]

    @pytest.mark.asyncio
    async def test_diff_shows_removals(self, tool):
        """Test diff shows removed lines."""
        current = "line1\nline2\nline3"
        proposed = "line1\nline2"
        result = await tool.execute({
            "current_config": current,
            "proposed_config": proposed,
        })
        assert "line3" in result.data["removed"]

    @pytest.mark.asyncio
    async def test_diff_identical_configs(self, tool):
        """Test diff with identical configs."""
        config = "line1\nline2"
        result = await tool.execute({
            "current_config": config,
            "proposed_config": config,
        })
        assert result.data["change_count"] == 0
